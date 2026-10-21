# src/proact/data/datasets/ego4d.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from proact.core.registry import DATASET_REGISTRY
from proact.data.datasets.utils.narrations import Narrations
from proact.utils.io import load_json


logging.basicConfig(level=logging.INFO)


def _resolve_long_video_path(ego4d_root: str, video_uid: str) -> str:
    return os.path.join(ego4d_root, "v2", "full_scale", f"{video_uid}.mp4")


def _resolve_annotation_path(root: str, version: str, task_json: str) -> str:
    if os.path.isabs(task_json):
        return task_json
    return os.path.join(root, version, "annotations", task_json)


def _load_narrations(ego4d_root: str) -> Narrations:
    return Narrations(os.path.join(ego4d_root, "v2", "annotations"))


def _ensure_grouped_annotation(data: Any, task_json: str) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise TypeError(
            f"Grouped ProactBench annotation '{task_json}' must load into a mapping of "
            f"video_uid -> group, got {type(data).__name__}."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for video_uid, group in data.items():
        if not isinstance(video_uid, str):
            raise TypeError(
                "Grouped ProactBench annotations require string top-level keys "
                f"(got {type(video_uid).__name__})."
            )
        if not isinstance(group, dict):
            raise TypeError(
                f"Group entry for video_uid='{video_uid}' must be a dict, got {type(group).__name__}."
            )
        matches = group.get("matches")
        if not isinstance(matches, list):
            raise TypeError(
                f"Group entry for video_uid='{video_uid}' must contain a list under 'matches'."
            )
        normalized[video_uid] = group

    return normalized


@DATASET_REGISTRY.register("ProactBench")
class ProactBench(Dataset):
    def __init__(self, root, **params):
        """
        root: this is the Ego4D dataset path (top-most folder)
        """
        self.root = root
        self.version = "v0" if params.get("version") is None else params.get("version")

        # 1. This is ideally json with ALL data for ALL tasks
        #    We will use this as index since it returns annotations in list of dict formats
        self.meta = load_json(_resolve_annotation_path(root, self.version, params["task_json"]))  #  this should be meta data about the videos

        # 2. We wil also load other data like narrations that CAN be used :)
        #    To this end we use our goalstep / narration / meta handler ;)
        self.narrations = _load_narrations(params["ego4d_root"])

        self.params = params

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        # return a dict; keep keys stable across datasets
        entry = dict(self.meta[i])

        # 1. We attach a narrations object (can be used to get narrations in different format at runtime)
        entry["narrations_object"] = self.narrations
        entry["video_path"] = _resolve_long_video_path(
            self.params["ego4d_root"], entry["video_uid"]
        )

        return entry


@DATASET_REGISTRY.register("ProactBenchGrouped")
class ProactBenchGrouped(Dataset):
    """
    Grouped ProactBench view for evaluation.

    One dataset item corresponds to one video and contains all GT matches for that
    video so long-video evaluation can reuse a single full-video execution trace.
    """

    def __init__(self, root, **params):
        self.root = root
        self.params = params
        self.version = "v0" if params.get("version") is None else params.get("version")
        self.task_json = params["task_json"]
        self.ego4d_root = params["ego4d_root"]
        self.narrations = _load_narrations(self.ego4d_root)

        annotation_path = _resolve_annotation_path(root, self.version, self.task_json)
        grouped = _ensure_grouped_annotation(load_json(annotation_path), self.task_json)
        self.meta = self._listify_groups(grouped)

    def _listify_groups(self, grouped: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []

        for video_uid in sorted(grouped.keys()):
            raw_group = grouped[video_uid]
            matches = raw_group["matches"]
            group_metadata = {
                key: value for key, value in raw_group.items() if key != "matches"
            }

            normalized_matches: list[dict[str, Any]] = []
            for match_index, raw_match in enumerate(matches):
                if not isinstance(raw_match, dict):
                    raise TypeError(
                        f"Match {match_index} for video_uid='{video_uid}' must be a dict, "
                        f"got {type(raw_match).__name__}."
                    )

                match = dict(raw_match)
                match_video_uid = match.get("video_uid", video_uid)
                if match_video_uid != video_uid:
                    raise ValueError(
                        f"Match {match_index} declares video_uid='{match_video_uid}' but is nested "
                        f"under top-level video_uid='{video_uid}'."
                    )

                match["video_uid"] = video_uid
                match["match_index"] = match_index
                match["match_id"] = f"{video_uid}:{match_index}"
                normalized_matches.append(match)

            group_entry = {
                "video_uid": video_uid,
                "video_path": _resolve_long_video_path(self.ego4d_root, video_uid),
                "matches": normalized_matches,
                "group_metadata": group_metadata,
                "narrations_object": self.narrations,
            }
            for key, value in group_metadata.items():
                if key not in group_entry:
                    group_entry[key] = value

            groups.append(group_entry)

        return groups

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        entry = dict(self.meta[i])
        entry["matches"] = [dict(match) for match in self.meta[i]["matches"]]
        entry["group_metadata"] = dict(self.meta[i]["group_metadata"])
        entry["narrations_object"] = self.narrations
        return entry


@DATASET_REGISTRY.register("ProactBenchClips")
class ProactBenchClips(Dataset):
    """Annotation-independent clip catalog for trace generation."""

    def __init__(self, root, **params):
        self.root = str(root)
        self.params = dict(params)
        task_json = params["task_json"]
        annotation_path = Path(task_json)
        if not annotation_path.is_absolute():
            annotation_path = Path(root) / task_json

        payload = load_json(str(annotation_path))
        if not isinstance(payload, dict):
            raise TypeError("Canonical ProactBench input must be a JSON object.")
        videos = payload.get("videos")
        clip_index = payload.get("clip_index")
        if not isinstance(videos, dict) or not isinstance(clip_index, dict):
            raise TypeError(
                "Canonical ProactBench input requires object-valued 'videos' and 'clip_index'."
            )

        self.annotation_path = str(annotation_path)
        self.schema_version = payload.get("schema_version")
        self.dataset_version = payload.get("dataset_version")
        self.task = payload.get("task")
        self.meta = self._build_clip_catalog(clip_index)

    @staticmethod
    def _build_clip_catalog(clip_index: dict[str, Any]) -> list[dict[str, Any]]:
        clips: list[dict[str, Any]] = []
        for clip_uid in sorted(clip_index):
            raw_clip = clip_index[clip_uid]
            if not isinstance(raw_clip, dict):
                raise TypeError(
                    f"clip_index['{clip_uid}'] must be an object, got {type(raw_clip).__name__}."
                )
            declared_uid = raw_clip.get("clip_uid", clip_uid)
            if declared_uid != clip_uid:
                raise ValueError(
                    f"clip_index key '{clip_uid}' disagrees with clip_uid='{declared_uid}'."
                )
            video_uid = raw_clip.get("video_uid") or raw_clip.get("source_video_uid")
            clip_path = raw_clip.get("clip_path")
            duration = raw_clip.get("clip_duration_sec")
            if not isinstance(video_uid, str) or not video_uid:
                raise ValueError(f"Clip '{clip_uid}' is missing video_uid.")
            if not isinstance(clip_path, str) or not clip_path:
                raise ValueError(f"Clip '{clip_uid}' is missing clip_path.")
            try:
                duration_sec = float(duration)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Clip '{clip_uid}' has invalid clip_duration_sec={duration!r}."
                ) from exc
            if duration_sec < 0:
                raise ValueError(f"Clip '{clip_uid}' has negative duration.")

            clips.append(
                {
                    "clip_uid": clip_uid,
                    "video_uid": video_uid,
                    "clip_path": clip_path,
                    "clip_duration_sec": duration_sec,
                    "clip_kind": raw_clip.get("clip_kind"),
                }
            )
        return clips

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        return dict(self.meta[i])


def _resolve_science_video_path(video_root: str, video_uid: str) -> str:
    return os.path.join(video_root, f"{video_uid}.mp4")


def _probe_video_duration_sec(path: str) -> float | None:
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


DEFAULT_FINEBIO_VIDEO_ROOT = os.environ.get("PROACT_SCIENCE_VIDEO_ROOT", "")


@DATASET_REGISTRY.register("ProactBenchScienceClips")
class ProactBenchScienceClips(Dataset):
    """Clip catalog for the science track (proact-studio-v3 schema).

    Handles two studio-v3 shapes:

    * Pre-cut clips (e.g. video_capture): a non-empty top-level ``clips`` dict
      is used verbatim — each entry already carries clip_path/duration_sec for a
      rendered clip. This mirrors the main ``ProactBenchClips`` contract.
    * Full-video catalogs (e.g. state_tracking, object_tracking): ``clips`` is
      empty/absent, so each source video in ``videos`` is evaluated as a single
      full-video clip resolving to <video_root>/<video_uid>.mp4.

    In both cases clip paths can be overridden at runtime with --clip-root, and
    only the fields the generation pipeline consumes are emitted.
    """

    def __init__(self, root, **params):
        self.root = str(root)
        self.params = dict(params)
        task_json = params["task_json"]
        annotation_path = Path(task_json)
        if not annotation_path.is_absolute():
            annotation_path = Path(root) / task_json

        payload = load_json(str(annotation_path))
        if not isinstance(payload, dict):
            raise TypeError("Science ProactBench input must be a JSON object.")

        self.annotation_path = str(annotation_path)
        self.schema_version = payload.get("schema_version")
        self.task = payload.get("task")
        video_root = params.get("video_root") or DEFAULT_FINEBIO_VIDEO_ROOT
        self.video_root = str(video_root)

        clips = payload.get("clips")
        if isinstance(clips, dict) and clips:
            self.meta = self._build_from_clips(clips)
        else:
            videos = payload.get("videos")
            if not isinstance(videos, dict) or not videos:
                raise TypeError(
                    "Science ProactBench input requires a non-empty object-valued "
                    "'videos' (or a non-empty 'clips' catalog)."
                )
            self.meta = self._build_clip_catalog(videos)

    def _build_from_clips(self, clips: dict[str, Any]) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for clip_uid in sorted(clips):
            entry = clips[clip_uid]
            if not isinstance(entry, dict):
                raise TypeError(
                    f"clips['{clip_uid}'] must be an object, got {type(entry).__name__}."
                )
            declared = entry.get("clip_uid", clip_uid)
            if declared != clip_uid:
                raise ValueError(
                    f"clips key '{clip_uid}' disagrees with clip_uid='{declared}'."
                )
            video_uid = entry.get("video_uid") or clip_uid
            clip_path = entry.get("clip_path")
            if not isinstance(clip_path, str) or not clip_path:
                clip_path = _resolve_science_video_path(self.video_root, clip_uid)
            duration = entry.get("clip_duration_sec")
            if duration is None:
                duration = entry.get("duration_sec")
            try:
                duration_sec = float(duration)
            except (TypeError, ValueError):
                duration_sec = _probe_video_duration_sec(clip_path) or 0.0
            if duration_sec < 0:
                raise ValueError(f"Clip '{clip_uid}' has negative duration.")
            catalog.append(
                {
                    "clip_uid": clip_uid,
                    "video_uid": video_uid,
                    "clip_path": clip_path,
                    "clip_duration_sec": duration_sec,
                    "clip_kind": entry.get("clip_kind", "clip"),
                }
            )
        return catalog

    def _build_clip_catalog(self, videos: dict[str, Any]) -> list[dict[str, Any]]:
        clips: list[dict[str, Any]] = []
        for video_uid in sorted(videos):
            entry = videos[video_uid]
            if not isinstance(entry, dict):
                raise TypeError(
                    f"videos['{video_uid}'] must be an object, got {type(entry).__name__}."
                )
            declared = entry.get("video_uid", video_uid)
            if declared != video_uid:
                raise ValueError(
                    f"videos key '{video_uid}' disagrees with video_uid='{declared}'."
                )
            clip_path = _resolve_science_video_path(self.video_root, video_uid)
            duration = _probe_video_duration_sec(clip_path)
            if duration is None or duration <= 0:
                sample_ends = [
                    float((s.get("times") or {}).get("end_sec", 0.0) or 0.0)
                    for s in (entry.get("samples") or {}).values()
                    if isinstance(s, dict)
                ]
                duration = (max(sample_ends) + 5.0) if sample_ends else 0.0
            clips.append(
                {
                    "clip_uid": video_uid,
                    "video_uid": video_uid,
                    "clip_path": clip_path,
                    "clip_duration_sec": float(duration),
                    "clip_kind": "full",
                }
            )
        return clips

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        return dict(self.meta[i])


if __name__ == "__main__":
    pb = ProactBench(
        "/path/to/proact-benchmark/data/proact_bench",
        version="v0",
        )
    print(pb[0])
