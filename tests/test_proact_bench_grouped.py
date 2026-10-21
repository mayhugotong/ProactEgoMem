from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support import install_common_stubs


install_common_stubs()

from proact.data.datasets.proact_bench import ProactBenchGrouped


def _write_fake_grouped_environment(base: Path, annotation_payload) -> tuple[Path, Path]:
    dataset_root = base / "dataset"
    ego4d_root = base / "ego4d"

    (dataset_root / "v1" / "annotations").mkdir(parents=True, exist_ok=True)
    (ego4d_root / "v2" / "annotations").mkdir(parents=True, exist_ok=True)
    (ego4d_root / "v2" / "full_scale").mkdir(parents=True, exist_ok=True)

    (dataset_root / "v1" / "annotations" / "task.json").write_text(
        json.dumps(annotation_payload),
        encoding="utf-8",
    )
    (ego4d_root / "v2" / "annotations" / "narration.json").write_text("{}", encoding="utf-8")
    video_ids = sorted(annotation_payload.keys()) if isinstance(annotation_payload, dict) else ["vid-a", "vid-b"]
    (ego4d_root / "ego4d.json").write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_uid": video_uid,
                        "is_grouped": False,
                        "group_video_uids": None,
                        "video_metadata": {"video_duration_sec": 120.0},
                    }
                    for video_uid in video_ids
                ]
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, ego4d_root


class ProactBenchGroupedTests(unittest.TestCase):
    def test_grouped_dataset_returns_one_item_per_video(self) -> None:
        annotation_payload = {
            "vid-b": {
                "source": "manual",
                "matches": [
                    {"video_uid": "vid-b", "start_time_sec": 5.0, "end_time_sec": 9.0, "object": "cup"}
                ],
            },
            "vid-a": {
                "matches": [
                    {"start_time_sec": 10.0, "end_time_sec": 20.0, "object": "map"},
                    {"video_uid": "vid-a", "start_time_sec": 40.0, "end_time_sec": 44.0, "object": "keys"},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root, ego4d_root = _write_fake_grouped_environment(Path(tmp_dir), annotation_payload)
            dataset = ProactBenchGrouped(
                root=str(dataset_root),
                version="v1",
                task_json="task.json",
                ego4d_root=str(ego4d_root),
            )

            self.assertEqual(len(dataset), 2)
            first = dataset[0]

            self.assertEqual(first["video_uid"], "vid-a")
            self.assertEqual(
                first["video_path"],
                str(ego4d_root / "v2" / "full_scale" / "vid-a.mp4"),
            )
            self.assertEqual(len(first["matches"]), 2)
            self.assertEqual(first["matches"][0]["match_index"], 0)
            self.assertEqual(first["matches"][0]["match_id"], "vid-a:0")
            self.assertEqual(first["matches"][1]["match_id"], "vid-a:1")
            self.assertIn("narrations_object", first)

    def test_grouped_dataset_rejects_non_mapping_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root, ego4d_root = _write_fake_grouped_environment(Path(tmp_dir), [{"video_uid": "vid-a"}])
            with self.assertRaisesRegex(TypeError, "must load into a mapping"):
                ProactBenchGrouped(
                    root=str(dataset_root),
                    version="v1",
                    task_json="task.json",
                    ego4d_root=str(ego4d_root),
                )


if __name__ == "__main__":
    unittest.main()
