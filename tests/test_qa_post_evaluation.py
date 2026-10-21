from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import install_common_stubs

install_common_stubs()

from proact.evaluation.clip_artifacts import sha256_file
from proact.evaluation.io.serialization import write_json
from proact.evaluation.post_evaluation.qa import (
    _adapt_studio_v3_payload,
    _clip_variant_plan,
    _extract_token_usage,
    _normalize_qa_item,
    _remap_ot_note,
    _shuffle_choices_deterministic,
    _sample_interval_frames,
    ClipFrameStore,
    allocate_frame_budget,
    build_compact_segments,
    build_oracle_segments,
    evaluate_qa_traces,
    load_qa_items,
    parse_answer_letter,
    parse_answer_response,
    sample_uniform_clip_frames,
    sample_uniform_recording_frames,
)


def qa_payload() -> dict:
    return {
        "schema_version": "test",
        "task": "state_tracking",
        "clip_index": {
            "clip-a": {
                "clip_uid": "clip-a",
                "clip_duration_sec": 12.0,
                "qa_items": {
                    "qa-1": {
                        "qa_id": "qa-1",
                        "question": "What was tracked?",
                        "answer": 2,
                        "answer_choices": [
                            {"choice_id": "choice-a", "text": "A thing"},
                            {"choice_id": "choice-b", "text": 2},
                            {"choice_id": "choice-c", "text": "Nothing"},
                            {"choice_id": "choice-d", "text": "Unknown"},
                        ],
                        "correct_choice_id": "choice-b",
                    }
                },
            }
        },
    }


def trace_payload(*, clip_uid: str = "clip-a") -> dict:
    return {
        "trace_schema_version": "proact.clip-trace.v2",
        "status": "complete",
        "run_fingerprint": "run-fp",
        "clip": {
            "clip_uid": clip_uid,
            "video_uid": "video-a",
            "clip_duration_sec": 12.0,
            "input_video_path": "/missing/clip-a.mp4",
        },
        "trace": {
            "trace_schema_version": "proact.clip-trace.v2",
            "predicted_intervals": [
                {
                    "interval_id": "reminder-2",
                    "instance_id": "reminder-2",
                    "interval_type": "reminder",
                    "tool_start_name": "start_reminder",
                    "tool_end_name": "stop_reminder",
                    "args": {"name": "faucet", "state": "running", "id": "reminder-2"},
                    "predicted_start_time": 1.0,
                    "predicted_end_time": 4.0,
                    "start_step_index": 1,
                    "end_step_index": 4,
                    "is_open": False,
                },
                {
                    "interval_id": "reminder-3",
                    "instance_id": "reminder-3",
                    "interval_type": "reminder",
                    "tool_start_name": "start_reminder",
                    "tool_end_name": None,
                    "args": {"name": "stove", "state": "on"},
                    "predicted_start_time": 5.0,
                    "predicted_end_time": None,
                    "start_step_index": 5,
                    "end_step_index": None,
                    "is_open": True,
                },
            ],
            "output_events": [],
        },
    }


def fake_clip_frames(seconds: int) -> list:
    """One distinct frame per second, as the 1-fps decode grid would yield."""
    import numpy as np

    return [
        SimpleNamespace(
            array=np.full((4, 4, 3), t, dtype=np.uint8),
            time=float(t),
            chosen_for_time=float(t),
            frame_index=t,
            pts=None,
        )
        for t in range(seconds)
    ]


def fake_store(seconds: int):
    config = SimpleNamespace(
        frame_sampling=SimpleNamespace(
            include_endpoints=True, unique_frames=True, convert_to_rgb=True,
            precise_seek=True, resize_frames=False, resize_width=None,
            resize_height=None, backend="pyav-cpu", ffmpeg_path=None,
        )
    )
    with patch(
        "proact.utils.media.sample_frames_at_fps_with_metadata",
        return_value=(fake_clip_frames(seconds), {"selected_backend": "fake"}),
    ):
        return ClipFrameStore(video_path="clip.mp4", config=config)


def recording(segment_id: str, start: float, end: float) -> dict:
    return {"segment_id": segment_id, "start_tool_name": "start_recording",
            "started_at_sec": start, "evidence_end_sec": end}


class FakeLLM:
    def __init__(self, text: str = "B") -> None:
        self.text = text
        self.calls = 0

    def complete_one(self, request):
        self.calls += 1
        return SimpleNamespace(
            ok=True,
            text=self.text,
            error=None,
            error_type=None,
            status_code=None,
            finish_reason="stop",
            reasoning=None,
            raw={},
        )


class QALoadingTests(unittest.TestCase):
    def test_load_qa_items_accepts_mapping_and_labels_choices(self):
        items = load_qa_items(qa_payload())
        qa = items["clip-a"][0]

        self.assertEqual(qa.qa_id, "qa-1")
        self.assertEqual(qa.answer, "2")
        # Choices are deterministically reordered; assert content + correct mapping
        # rather than a fixed letter position.
        self.assertEqual([c["letter"] for c in qa.answer_choices], ["A", "B", "C", "D"])
        self.assertEqual(
            sorted(str(c["text"]) for c in qa.answer_choices),
            sorted(["A thing", "2", "Nothing", "Unknown"]),
        )
        correct = [c for c in qa.answer_choices if c["letter"] == qa.correct_letter]
        self.assertEqual(len(correct), 1)
        self.assertEqual(correct[0]["choice_id"], "choice-b")
        self.assertEqual(str(correct[0]["text"]), "2")


class CompactSegmentTests(unittest.TestCase):
    def test_compact_segments_render_overlay_as_reminder_and_close_open_at_duration(self):
        compact = build_compact_segments(
            trace_payload(),
            data_clip={"clip_duration_sec": 12.0},
            task_name="state_tracking",
        )
        first, second = compact["segments"]

        self.assertEqual(first["segment_id"], "reminder-2")
        self.assertEqual(first["instance_id"], "reminder-2")
        self.assertEqual(first["interval_type"], "reminder")
        self.assertEqual(first["start_tool_name"], "start_reminder")
        self.assertEqual(first["end_tool_name"], "stop_reminder")
        self.assertEqual(first["arguments"]["id"], "reminder-2")
        self.assertEqual(first["evidence_end_sec"], 4.0)
        self.assertEqual(second["status"], "open")
        self.assertIsNone(second["ended_at_sec"])
        self.assertEqual(second["evidence_end_sec"], 12.0)


class FrameBudgetTests(unittest.TestCase):
    def test_allocate_frame_budget_balances_and_truncates(self):
        segments = [
            {"segment_id": f"s{idx}", "start_tool_name": "start_recording", "started_at_sec": idx}
            for idx in range(5)
        ]

        balanced = allocate_frame_budget(segments[:3], 8)
        self.assertEqual([item["frames"] for item in balanced["allocations"]], [3, 3, 2])
        self.assertFalse(balanced["truncated"])

        truncated = allocate_frame_budget(segments, 3)
        self.assertEqual([item["segment_id"] for item in truncated["allocations"]], ["s0", "s1", "s2"])
        self.assertEqual([item["frames"] for item in truncated["allocations"]], [1, 1, 1])
        self.assertTrue(truncated["truncated"])
        self.assertEqual(truncated["omitted_segment_ids"], ["s3", "s4"])

    def test_memory_frames_are_uniform_over_the_recording_union(self):
        store = fake_store(180)
        frames, alloc = sample_uniform_recording_frames(
            store=store,
            segments=[recording("r1", 10.0, 20.0), recording("r2", 100.0, 130.0)],
            frame_budget=40,
            clip_duration_sec=180.0,
        )
        times = [frame.timestamp_sec for frame in frames]
        # 40 captured seconds, 40 frames: one per captured second, so the 10 s
        # recording gets 10 frames and the 30 s one gets 30.
        self.assertEqual(times, [float(t) for t in list(range(10, 20)) + list(range(100, 130))])
        self.assertEqual([f.segment_id for f in frames].count("r1"), 10)
        self.assertEqual(alloc["covered_sec"], 40.0)
        self.assertEqual(alloc["served_frames"], 40)
        self.assertFalse(alloc["truncated"])

    def test_memory_frames_merge_overlaps_and_ignore_other_tools(self):
        store = fake_store(180)
        frames, alloc = sample_uniform_recording_frames(
            store=store,
            segments=[
                recording("r1", 10.0, 30.0),
                recording("r2", 20.0, 40.0),
                {"segment_id": "x", "start_tool_name": "start_state_tracking",
                 "started_at_sec": 150.0, "evidence_end_sec": 170.0},
            ],
            frame_budget=3,
            clip_duration_sec=180.0,
        )
        self.assertEqual(alloc["union_intervals"], [[10.0, 40.0]])
        self.assertEqual([frame.timestamp_sec for frame in frames], [15.0, 25.0, 35.0])

    def test_memory_frames_short_union_serves_fewer_and_says_so(self):
        store = fake_store(180)
        frames, alloc = sample_uniform_recording_frames(
            store=store, segments=[recording("r1", 5.0, 8.0)],
            frame_budget=40, clip_duration_sec=180.0,
        )
        self.assertEqual([frame.timestamp_sec for frame in frames], [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(alloc["served_frames"], 4)
        self.assertTrue(alloc["truncated"])

    def test_memory_frames_empty_without_recordings(self):
        frames, alloc = sample_uniform_recording_frames(
            store=fake_store(180), segments=[], frame_budget=40, clip_duration_sec=180.0,
        )
        self.assertEqual(frames, [])
        self.assertEqual(alloc["served_frames"], 0)

    def test_baseline_frames_are_uniform_over_the_whole_clip(self):
        frames, alloc = sample_uniform_clip_frames(
            store=fake_store(180), clip_duration_sec=180.0, frame_budget=40,
        )
        times = [frame.timestamp_sec for frame in frames]
        self.assertEqual(len(times), 40)
        self.assertEqual(len(set(times)), 40)
        self.assertEqual((times[0], times[-1]), (0.0, 179.0))
        self.assertEqual(alloc["allocation_policy"], "uniform_whole_clip")

    def test_single_frame_interval_uses_one_target_cuda_window(self):
        config = SimpleNamespace(
            frame_sampling=SimpleNamespace(
                include_endpoints=True,
                unique_frames=True,
                convert_to_rgb=True,
                precise_seek=True,
                resize_frames=True,
                resize_width=336,
                resize_height=336,
                backend="ffmpeg-cuda",
                ffmpeg_path="/ffmpeg",
            )
        )
        fake_frame = SimpleNamespace(
            array=[[[1, 2, 3]]],
            time=5.0,
            chosen_for_time=5.0,
            frame_index=0,
            pts=None,
        )

        with patch(
            "proact.utils.media.sample_frames_at_fps_with_metadata",
            return_value=(
                [fake_frame],
                {
                    "selected_backend": "ffmpeg-cuda",
                    "processing_path": "cuda-decode+cuda-scale+cpu-fps",
                    "fallback_reason": None,
                },
            ),
        ) as sampler:
            frames = _sample_interval_frames(
                video_path="clip.mp4",
                segment={
                    "segment_id": "recording-1",
                    "started_at_sec": 10.0,
                    "evidence_end_sec": 20.0,
                },
                count=1,
                config=config,
                adapter_anchor_frame=False,
            )

        self.assertEqual(len(frames), 1)
        kwargs = sampler.call_args.kwargs
        self.assertEqual(kwargs["backend"], "ffmpeg-cuda")
        self.assertFalse(kwargs["include_endpoints"])
        self.assertGreaterEqual(kwargs["S0"], 10.0)
        self.assertLessEqual(kwargs["S1"], 20.0)
        self.assertLessEqual(kwargs["S1"] - kwargs["S0"], 0.25)
        self.assertLess(kwargs["fps"] * (kwargs["S1"] - kwargs["S0"]), 1.0)


class AnswerParserTests(unittest.TestCase):
    def test_parse_answer_letter_is_strict(self):
        self.assertEqual(parse_answer_letter("A"), "A")
        self.assertEqual(parse_answer_letter(" b "), "B")
        self.assertIsNone(parse_answer_letter("Answer: A"))
        self.assertIsNone(parse_answer_letter("AB"))
        self.assertIsNone(parse_answer_letter(""))

    def test_parse_answer_response_uses_last_json_object(self):
        letter, payload, error = parse_answer_response(
            'draft {"reasoning": "first", "answer": "A"} final {"reasoning": "final", "answer": "b"}'
        )

        self.assertIsNone(error)
        self.assertEqual(letter, "B")
        self.assertEqual(payload["reasoning"], "final")

    def test_parse_answer_response_recovers_bare_letter_leniently(self):
        # Free-form/streaming answerers may emit just the letter; the lenient
        # fallback recovers it and flags the extraction.
        letter, payload, error = parse_answer_response("B")

        self.assertEqual(letter, "B")
        self.assertEqual(payload, {"answer": "B", "lenient_extraction": True})
        self.assertIsNone(error)


class QAEvaluationSmokeTests(unittest.TestCase):
    def test_stateless_success_and_resume_without_second_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt", encoding="utf-8")
            llms = root / "llms.yaml"
            llms.write_text(
                json.dumps(
                    {
                        "fake": {
                            "provider": "fake",
                            "model": "fake-model",
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            trace_dir = output / "shards" / "shard-00000-of-00001" / "clips" / "clip-a"
            trace_dir.mkdir(parents=True)
            trace = trace_payload()
            write_json(trace_dir / "trace.json", trace)
            write_json(
                trace_dir / "manifest.json",
                {
                    "status": "complete",
                    "clip_uid": "clip-a",
                    "trace_schema_version": "proact.clip-trace.v2",
                    "run_fingerprint": "run-fp",
                    "artifacts": {
                        "trace": {
                            "path": "trace.json",
                            "sha256": sha256_file(trace_dir / "trace.json"),
                        }
                    },
                },
            )
            write_json(output / "run_manifest.json", {"run_fingerprint": "run-fp"})
            data_json = root / "qa.json"
            write_json(data_json, qa_payload())
            eval_yaml = root / "eval.yaml"
            eval_yaml.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "dataset": "ProactBenchClips",
                            "root": str(root),
                            "split": "val",
                            "dataset_params": {"task_json": str(data_json)},
                        },
                        "context": "short-video",
                        "proactive_level": "intervention",
                        "input_mode": {
                            "mode": "sliding-window",
                            "fps": 1,
                            "window_size_frames": 1,
                            "stride_frames": 1,
                        },
                        "execution_backend": {"mode": "stateless"},
                        "task_policy": {
                            "task_name": "state_tracking",
                            "expected_interval_types": ["reminder"],
                            "expected_start_tools": ["start_reminder"],
                            "expected_end_tools": ["stop_reminder"],
                            "allowed_tool_names": ["start_reminder", "stop_reminder"],
                            "ignored_tool_names": [],
                        },
                        "prompt": {
                            "prompt_path": str(prompt),
                            "include_state": True,
                            "state_format": "compact",
                        },
                        "prompt_context": {
                            "include_active_tools": True,
                            "completed_tool_intervals": {"mode": "last_n", "n": 5},
                            "recent_tool_calls": {"mode": "none", "n": 5, "source": "executed"},
                            "recent_messages": {"mode": "none", "n": 10, "source": "executed"},
                        },
                        "parser": {
                            "unified_schema": True,
                            "allow_code_fence_json": False,
                        },
                        "frame_sampling": {
                            "backend": "pyav-cpu",
                            "allow_cpu_fallback": True,
                            "include_endpoints": True,
                            "unique_frames": True,
                            "convert_to_rgb": True,
                            "precise_seek": True,
                            "resize_frames": False,
                        },
                        "model": {
                            "llm_config_path": str(llms),
                            "profile": "fake",
                            "overrides": {},
                            "request_params": {},
                        },
                        "output": {
                            "output_dir": str(output),
                            "max_clip_attempts": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            # Correct choice is placed deterministically; answer with whatever letter
            # the loader assigns it so the "all correct" expectations still hold.
            correct_letter = load_qa_items(qa_payload())["clip-a"][0].correct_letter
            fake_llm = FakeLLM(
                json.dumps({"reasoning": "The compact memory tracks the faucet.", "answer": correct_letter})
            )

            # The no-memory baseline decodes the clip (frames uniform over the
            # whole clip), so hand it a fake 12-second clip on the 1-fps grid.
            with patch("proact.evaluation.post_evaluation.qa.build_llm_from_yaml", return_value=fake_llm), \
                    patch("proact.evaluation.post_evaluation.qa.resolve_clip_video_path",
                          return_value="/fake/clip-a.mp4"), \
                    patch("proact.utils.media.sample_frames_at_fps_with_metadata",
                          return_value=(fake_clip_frames(12), {"selected_backend": "fake"})):
                summary = evaluate_qa_traces(
                    data_json=data_json,
                    eval_config=eval_yaml,
                    frame_budget=4,
                )
                # Three variants per QA item: memory, oracle, baseline_no_memory.
                self.assertEqual(summary["answered"], 3)
                self.assertEqual(summary["failed"], 0)
                self.assertEqual(summary["expected"], 3)
                self.assertEqual(summary["correct"], 3)
                self.assertEqual(summary["answer_rate"], 1.0)
                self.assertEqual(summary["strict_accuracy_over_expected"], 1.0)
                self.assertEqual(summary["variants"]["memory"]["answered"], 1)
                self.assertEqual(summary["variants"]["memory"]["expected"], 1)
                self.assertEqual(summary["variants"]["memory"]["strict_accuracy_over_expected"], 1.0)
                self.assertEqual(summary["variants"]["baseline_no_memory"]["answered"], 1)
                self.assertEqual(summary["variants"]["baseline_no_memory"]["expected"], 1)
                self.assertEqual(summary["variants"]["baseline_no_memory"]["strict_accuracy_over_expected"], 1.0)
                self.assertEqual(summary["comparison"]["memory_minus_baseline_accuracy"], 0.0)
                self.assertEqual(summary["comparison"]["memory_minus_baseline_strict_accuracy"], 0.0)
                self.assertEqual(fake_llm.calls, 3)

                summary = evaluate_qa_traces(
                    data_json=data_json,
                    eval_config=eval_yaml,
                    frame_budget=4,
                )
                self.assertEqual(summary["answered"], 3)
                self.assertEqual(summary["reused"], 3)
                self.assertEqual(fake_llm.calls, 3)

            qa_record = json.loads((output / "qa" / "clips" / "clip-a" / "qa-1.json").read_text())
            self.assertEqual(qa_record["evaluation_variant"], "memory")
            self.assertEqual(qa_record["evidence_mode"], "compact_memory")
            self.assertEqual(qa_record["parsed_answer_letter"], correct_letter)
            self.assertEqual(qa_record["parsed_answer_payload"]["answer"], correct_letter)
            self.assertEqual(
                qa_record["debug_artifacts"],
                {
                    "prompt_text": "debug/clips/clip-a/qa-1.memory.prompt.txt",
                    "prompt_json": "debug/clips/clip-a/qa-1.memory.json",
                },
            )
            prompt_text = (output / "qa" / "debug" / "clips" / "clip-a" / "qa-1.memory.prompt.txt").read_text()
            self.assertIn("## Tool Semantics", prompt_text)
            self.assertIn('- [start_reminder: 1.000s, args: name="faucet", state="running"; stop_reminder: 4.000s]', prompt_text)
            self.assertIn(
                'Reply with one JSON object and nothing else: '
                '{"answer": "A" | "B" | "C" | "D"}',
                prompt_text,
            )
            self.assertIn("Pick the single best option", prompt_text)
            self.assertNotIn("## Evidence Use", prompt_text)
            self.assertIn("## Question\nWhat was tracked?", prompt_text)
            debug_json = json.loads((output / "qa" / "debug" / "clips" / "clip-a" / "qa-1.memory.json").read_text())
            self.assertEqual(debug_json["prompt_text"], prompt_text)
            self.assertEqual(debug_json["evaluation_variant"], "memory")
            self.assertEqual(debug_json["evidence_mode"], "compact_memory")

            baseline_record = json.loads(
                (output / "qa" / "clips" / "clip-a" / "qa-1.baseline_no_memory.json").read_text()
            )
            self.assertEqual(baseline_record["evaluation_variant"], "baseline_no_memory")
            self.assertEqual(baseline_record["evidence_mode"], "no_memory")
            self.assertEqual(baseline_record["compact_segments"], [])
            baseline_prompt = (
                output / "qa" / "debug" / "clips" / "clip-a" / "qa-1.baseline_no_memory.prompt.txt"
            ).read_text()
            self.assertIn("This is the no-memory baseline", baseline_prompt)
            self.assertIn("_No compact memory is provided for this baseline condition;", baseline_prompt)
            # 4 frames uniform over the whole 12 s clip, not text only.
            self.assertEqual(
                [frame["timestamp_sec"] for frame in baseline_record["sampled_frames"]],
                [0.0, 4.0, 8.0, 11.0],
            )
            self.assertIn("## Evidence Frames", baseline_prompt)
            # ST memory stays text: no frames attached.
            self.assertEqual(qa_record["sampled_frames"], [])

def v3_full_video_payload() -> dict:
    """proact-studio-v3 full-video shape (state/object tracking): no clips,
    QA + samples live under videos[video_uid]; clip_uid == video_uid."""
    return {
        "schema_version": "proact-studio-v3",
        "task": "object_tracking",
        "kind": "canonical",
        "clips": {},
        "videos": {
            "vid-1": {
                "video_uid": "vid-1",
                "samples": {
                    "s-1": {
                        "sample_id": "s-1",
                        "object": "micro_tube",
                        "object_description": "micro tube",
                        "note": "location_at_first_take: rack\nlocation_at_place: centrifuge\nhand_side: left",
                        "times": {"start_sec": 10.0, "end_sec": 14.0},
                        "clip": None,
                    }
                },
                "qa_items": {
                    "qa-1": {
                        "qa_id": "qa-1",
                        "question": "Where placed?",
                        "answer": "centrifuge",
                        "answer_choices": [
                            {"choice_id": "c0", "text": "rack"},
                            {"choice_id": "c1", "text": "centrifuge"},
                            {"choice_id": "c2", "text": "shelf"},
                            {"choice_id": "c3", "text": "bin"},
                        ],
                        "correct_choice_id": "c1",
                    }
                },
            }
        },
    }


def v3_precut_payload() -> dict:
    """proact-studio-v3 pre-cut shape (video_capture): QA under clips[clip_uid],
    samples carry a populated ``clip`` with clip-local times."""
    return {
        "schema_version": "proact-studio-v3",
        "task": "video_capture",
        "kind": "canonical",
        "clips": {
            "clip-x": {
                "clip_uid": "clip-x",
                "video_uid": "vid-9",
                "duration_sec": 180.0,
                "qa_items": {
                    "qa-9": {
                        "qa_id": "qa-9",
                        "question": "Setting?",
                        "answer": "37C",
                        "answer_choices": [
                            {"choice_id": "a", "text": "37C"},
                            {"choice_id": "b", "text": "36C"},
                            {"choice_id": "c", "text": "35C"},
                            {"choice_id": "d", "text": "34C"},
                        ],
                        "correct_choice_id": "a",
                    }
                },
            }
        },
        "videos": {
            "vid-9": {
                "video_uid": "vid-9",
                "samples": {
                    "s-9": {
                        "sample_id": "s-9",
                        "object": "incubator",
                        "object_description": "",
                        "note": "37C, 5% CO2",
                        "times": {"start_sec": 120.0, "end_sec": 124.0},
                        "clip": {"clip_uid": "clip-x", "local_start_sec": 0.0, "local_end_sec": 4.0},
                    }
                },
            }
        },
    }


class StudioV3AdapterTests(unittest.TestCase):
    def test_remap_ot_note_renames_location_keys(self):
        note = "location_at_first_take: rack\nlocation_at_place: bench\nhand_side: left"
        out = _remap_ot_note(note)
        self.assertIn("previous_location: rack", out)
        self.assertIn("new_location: bench", out)
        self.assertIn("hand_side: left", out)  # unmapped keys preserved

    def test_full_video_v3_loads_qa_and_builds_clip_index(self):
        payload = _adapt_studio_v3_payload(v3_full_video_payload())
        self.assertIn("clip_index", payload)
        self.assertIn("vid-1", payload["clip_index"])  # clip_uid == video_uid
        items = load_qa_items(payload)
        self.assertEqual([qa.qa_id for qa in items["vid-1"]], ["qa-1"])
        q = items["vid-1"][0]
        correct = [c for c in q.answer_choices if c["letter"] == q.correct_letter]
        self.assertEqual(correct[0]["choice_id"], "c1")  # placement-agnostic

    def test_precut_v3_loads_qa_from_clips(self):
        payload = _adapt_studio_v3_payload(v3_precut_payload())
        self.assertEqual(payload["clip_index"]["clip-x"]["clip_duration_sec"], 180.0)
        items = load_qa_items(payload)
        self.assertEqual([qa.qa_id for qa in items["clip-x"]], ["qa-9"])

    def test_oracle_segments_from_v3_full_video(self):
        payload = _adapt_studio_v3_payload(v3_full_video_payload())
        segs = build_oracle_segments(
            task_name="object_tracking",
            clip_uid="vid-1",
            clip_payload=payload["clip_index"]["vid-1"],
            videos_index=payload["videos"],
            clip_duration_sec=180.0,
        )
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["started_at_sec"], 10.0)
        self.assertEqual(segs[0]["ended_at_sec"], 14.0)
        self.assertEqual(segs[0]["arguments"]["name"], "micro_tube")
        self.assertEqual(segs[0]["arguments"]["previous_location"], "rack")
        self.assertEqual(segs[0]["arguments"]["new_location"], "centrifuge")

    def test_oracle_segments_from_v3_precut_clip(self):
        payload = _adapt_studio_v3_payload(v3_precut_payload())
        segs = build_oracle_segments(
            task_name="video_capture",
            clip_uid="clip-x",
            clip_payload=payload["clip_index"]["clip-x"],
            videos_index=payload["videos"],
            clip_duration_sec=180.0,
        )
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["started_at_sec"], 0.0)
        self.assertEqual(segs[0]["ended_at_sec"], 4.0)
        self.assertEqual(segs[0]["start_tool_name"], "start_recording")

    def test_choice_shuffle_is_deterministic_and_decorrelates_position(self):
        raw = [{"choice_id": f"c{i}", "text": str(i)} for i in range(4)]
        # deterministic per qa_id
        self.assertEqual(
            [c["choice_id"] for c in _shuffle_choices_deterministic("qa-x", raw)],
            [c["choice_id"] for c in _shuffle_choices_deterministic("qa-x", raw)],
        )
        # correct-always-first data must not map to a fixed letter across items
        import collections

        letters = collections.Counter()
        for k in range(400):
            item = {
                "qa_id": f"qa-{k}",
                "question": "q",
                "answer": "0",
                "correct_choice_id": "c0",  # always first in source
                "answer_choices": [{"choice_id": f"c{i}", "text": str(i)} for i in range(4)],
            }
            letters[_normalize_qa_item("clip", item).correct_letter] += 1
        # every letter should occur; none should dominate near-100% as in the raw data
        self.assertEqual(set(letters), {"A", "B", "C", "D"})
        self.assertLess(max(letters.values()), 0.4 * sum(letters.values()))

    def test_shuffle_preserves_correct_mapping(self):
        item = {
            "qa_id": "qa-keep",
            "question": "q",
            "answer": "right",
            "correct_choice_id": "c2",
            "answer_choices": [
                {"choice_id": "c0", "text": "w0"},
                {"choice_id": "c1", "text": "w1"},
                {"choice_id": "c2", "text": "right"},
                {"choice_id": "c3", "text": "w3"},
            ],
        }
        qa = _normalize_qa_item("clip", item)
        correct = [c for c in qa.answer_choices if c["letter"] == qa.correct_letter]
        self.assertEqual(len(correct), 1)
        self.assertEqual(correct[0]["choice_id"], "c2")
        self.assertEqual(correct[0]["text"], "right")

    def test_extract_token_usage_handles_providers(self):
        self.assertEqual(
            _extract_token_usage({"usage": {"input_tokens": 1200, "output_tokens": 40}}),
            {"input_tokens": 1200, "output_tokens": 40},
        )
        self.assertEqual(
            _extract_token_usage({"usageMetadata": {"promptTokenCount": 2600, "candidatesTokenCount": 15, "thoughtsTokenCount": 0}}),
            {"input_tokens": 2600, "output_tokens": 15},
        )
        self.assertEqual(
            _extract_token_usage({"usage": {"prompt_tokens": 900, "completion_tokens": 12}}),
            {"input_tokens": 900, "output_tokens": 12},
        )
        self.assertEqual(_extract_token_usage(None), {})
        self.assertEqual(_extract_token_usage({"nope": 1}), {})

    def test_variant_plan_has_no_uniform(self):
        # uniform (whole-clip) control was removed; every task = memory/oracle/baseline
        for mode in ("recording_frames", "compact_memory"):
            names = [v for v, _ in _clip_variant_plan(mode)]
            self.assertEqual(names, ["memory", "oracle", "baseline_no_memory"])
            self.assertNotIn("uniform", names)

    def test_load_qa_items_balances_correct_letter_exactly(self):
        # 40 items with the correct answer ALWAYS first (worst-case source bias).
        clip_index = {}
        for k in range(40):
            clip_index[f"clip-{k}"] = {
                "clip_uid": f"clip-{k}",
                "qa_items": {
                    "q": {
                        "qa_id": f"qa-{k}",
                        "question": "q",
                        "answer": "0",
                        "correct_choice_id": "c0",
                        "answer_choices": [{"choice_id": f"c{i}", "text": str(i)} for i in range(4)],
                    }
                },
            }
        items = load_qa_items({"schema_version": "test", "clip_index": clip_index})
        import collections

        dist = collections.Counter(q.correct_letter for qs in items.values() for q in qs)
        self.assertEqual(set(dist), {"A", "B", "C", "D"})
        self.assertLessEqual(max(dist.values()) - min(dist.values()), 1)  # exactly balanced

    def test_adapter_is_idempotent_and_ignores_legacy(self):
        legacy = qa_payload()  # already has clip_index, non-v3 schema
        self.assertIs(_adapt_studio_v3_payload(legacy), legacy)
        once = _adapt_studio_v3_payload(v3_full_video_payload())
        twice = _adapt_studio_v3_payload(once)
        self.assertEqual(load_qa_items(once).keys(), load_qa_items(twice).keys())


if __name__ == "__main__":
    unittest.main()
