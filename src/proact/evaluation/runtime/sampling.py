from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from proact.evaluation.config import FrameSamplingConfig, InputModeConfig


@dataclass
class SampleBatch:
    step_index: int
    timestamp_sec: float
    frames: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class DefaultVideoSampler:
    def __init__(self, frame_sampling: FrameSamplingConfig):
        self.frame_sampling = frame_sampling

    def generate_batches(self, video_path: str, input_mode: InputModeConfig) -> tuple[list[SampleBatch], dict[str, Any]]:
        if input_mode.mode == "full-video":
            return self._sample_full_video(video_path, input_mode)
        return self._sample_sliding_window(video_path, input_mode)

    def generate_stream_frames(
        self,
        video_path: str,
        *,
        fps: float,
    ) -> tuple[list[Any], dict[str, Any]]:
        from proact.utils.media import sample_frames_at_fps_with_metadata

        sampled_frames, backend_metadata = sample_frames_at_fps_with_metadata(
            path=video_path,
            fps=float(fps),
            include_endpoints=self.frame_sampling.include_endpoints,
            unique=self.frame_sampling.unique_frames,
            convert_to_rgb=self.frame_sampling.convert_to_rgb,
            precise_seek=self.frame_sampling.precise_seek,
            resize_width=int(self.frame_sampling.resize_width or 0) if self.frame_sampling.resize_frames else None,
            resize_height=int(self.frame_sampling.resize_height or 0) if self.frame_sampling.resize_frames else None,
            backend=self.frame_sampling.backend,
            ffmpeg_path=self.frame_sampling.ffmpeg_path,
        )
        sampled_frames = self._prepare_frames(sampled_frames)
        sampled_frame_fingerprints = self._frame_fingerprints(sampled_frames)

        metadata = {
            "sampling_mode": "native-streaming",
            "input_video_path": video_path,
            "fps": float(fps),
            "sampled_frame_count_total": len(sampled_frames),
            "sampled_frame_times": [float(frame.time) for frame in sampled_frames],
            "sampled_frame_fingerprints": sampled_frame_fingerprints,
            "frame_resize": self._frame_resize_metadata(),
            "frame_memory": {"owned": True, "contiguous": True},
            "backend": backend_metadata,
        }
        return sampled_frames, metadata

    def _sample_sliding_window(
        self,
        video_path: str,
        input_mode: InputModeConfig,
    ) -> tuple[list[SampleBatch], dict[str, Any]]:
        from proact.utils.media import sample_frames_at_fps_with_metadata

        sampled_frames, backend_metadata = sample_frames_at_fps_with_metadata(
            path=video_path,
            fps=float(input_mode.fps),
            include_endpoints=self.frame_sampling.include_endpoints,
            unique=self.frame_sampling.unique_frames,
            convert_to_rgb=self.frame_sampling.convert_to_rgb,
            precise_seek=self.frame_sampling.precise_seek,
            resize_width=int(self.frame_sampling.resize_width or 0) if self.frame_sampling.resize_frames else None,
            resize_height=int(self.frame_sampling.resize_height or 0) if self.frame_sampling.resize_frames else None,
            backend=self.frame_sampling.backend,
            ffmpeg_path=self.frame_sampling.ffmpeg_path,
        )
        sampled_frames = self._prepare_frames(sampled_frames)
        sampled_frame_fingerprints = self._frame_fingerprints(sampled_frames)

        stride_frames = int(input_mode.stride_frames or 0)
        window_size_frames = int(input_mode.window_size_frames or 0)
        end_indices = list(range(stride_frames, len(sampled_frames) + 1, stride_frames))
        if not end_indices or end_indices[-1] != len(sampled_frames):
            end_indices.append(len(sampled_frames))

        batches: list[SampleBatch] = []
        for step_index, end_index in enumerate(end_indices):
            start_index = max(0, end_index - window_size_frames)
            frames = sampled_frames[start_index:end_index]
            batches.append(
                SampleBatch(
                    step_index=step_index,
                    timestamp_sec=float(frames[-1].time),
                    frames=frames,
                    metadata={
                        "start_sample_index": start_index,
                        "end_sample_index": end_index - 1,
                        "sampled_frame_fingerprints": sampled_frame_fingerprints[start_index:end_index],
                    },
                )
            )

        metadata = {
            "sampling_mode": input_mode.mode,
            "input_video_path": video_path,
            "fps": float(input_mode.fps),
            "window_size_frames": window_size_frames,
            "stride_frames": stride_frames,
            "sampled_frame_count_total": len(sampled_frames),
            "step_count": len(batches),
            "sampled_frame_times": [float(frame.time) for frame in sampled_frames],
            "sampled_frame_fingerprints": sampled_frame_fingerprints,
            "frame_resize": self._frame_resize_metadata(),
            "frame_memory": {"owned": True, "contiguous": True},
            "backend": backend_metadata,
        }
        return batches, metadata

    def _sample_full_video(
        self,
        video_path: str,
        input_mode: InputModeConfig,
    ) -> tuple[list[SampleBatch], dict[str, Any]]:
        if self.frame_sampling.backend == "ffmpeg-cuda":
            raise ValueError(
                "frame_sampling.backend=ffmpeg-cuda is not supported for full-video "
                "uniform sampling; use sliding-window/chunked-window or set "
                "backend=pyav-cpu explicitly."
            )
        from proact.utils.media import sample_uniform_frames_memory_friendly

        frames = sample_uniform_frames_memory_friendly(
            path=video_path,
            N=int(input_mode.num_frames or 0),
            include_endpoints=self.frame_sampling.include_endpoints,
            unique=self.frame_sampling.unique_frames,
            convert_to_rgb=self.frame_sampling.convert_to_rgb,
            precise_seek=self.frame_sampling.precise_seek,
        )
        frames = self._prepare_frames(frames)
        sampled_frame_fingerprints = self._frame_fingerprints(frames)

        batch = SampleBatch(
            step_index=0,
            timestamp_sec=float(frames[-1].time) if frames else 0.0,
            frames=frames,
            metadata={
                "start_sample_index": 0,
                "end_sample_index": max(0, len(frames) - 1),
                "sampled_frame_fingerprints": sampled_frame_fingerprints,
            },
        )
        metadata = {
            "sampling_mode": input_mode.mode,
            "input_video_path": video_path,
            "num_frames": int(input_mode.num_frames or 0),
            "sampled_frame_count_total": len(frames),
            "step_count": 1,
            "sampled_frame_times": [float(frame.time) for frame in frames],
            "sampled_frame_fingerprints": sampled_frame_fingerprints,
            "frame_resize": self._frame_resize_metadata(),
            "frame_memory": {"owned": True, "contiguous": True},
            "backend": {
                "requested_backend": self.frame_sampling.backend,
                "selected_backend": "pyav-cpu",
                "decoder": "pyav",
                "ffmpeg_path": self.frame_sampling.ffmpeg_path,
                "processing_path": "pyav-two-pass",
                "gpu_resize": False,
            },
        }
        return [batch], metadata

    def _prepare_frames(self, frames: list[Any]) -> list[Any]:
        frames = self._resize_frames_if_needed(frames)
        import numpy as np

        previous_time: float | None = None
        for index, frame in enumerate(frames):
            timestamp = float(frame.time)
            if previous_time is not None and timestamp < previous_time:
                raise ValueError(
                    f"Sampled frame timestamps are not monotonic at index {index}: "
                    f"{timestamp} < {previous_time}."
                )
            previous_time = timestamp

            array = np.asarray(getattr(frame, "array", None))
            if array.ndim not in (2, 3) or array.size == 0:
                raise ValueError(
                    f"Sampled frame {index} has invalid shape {getattr(array, 'shape', None)}."
                )
            owned = np.ascontiguousarray(array).copy()
            if not owned.flags.c_contiguous or not owned.flags.owndata:
                raise RuntimeError(f"Sampled frame {index} does not own contiguous memory.")
            frame.array = owned
        return frames

    def _resize_frames_if_needed(self, frames: list[Any]) -> list[Any]:
        if not self.frame_sampling.resize_frames:
            return frames

        from proact.utils.media import resize_frame

        width = int(self.frame_sampling.resize_width or 0)
        height = int(self.frame_sampling.resize_height or 0)
        for frame in frames:
            frame.array = resize_frame(frame.array, width=width, height=height)
        return frames

    def _frame_resize_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.frame_sampling.resize_frames,
            "width": self.frame_sampling.resize_width,
            "height": self.frame_sampling.resize_height,
        }

    @staticmethod
    def _frame_fingerprints(frames: list[Any]) -> list[str | None]:
        fingerprints: list[str | None] = []
        for frame in frames:
            array = getattr(frame, "array", None)
            shape = getattr(array, "shape", None)
            dtype = getattr(array, "dtype", None)
            if shape is None or dtype is None or not hasattr(array, "tobytes"):
                fingerprints.append(None)
                continue
            try:
                digest = hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest()
            except Exception:
                fingerprints.append(None)
                continue
            fingerprints.append(f"{digest}:{tuple(shape)}:{dtype}")
        return fingerprints
