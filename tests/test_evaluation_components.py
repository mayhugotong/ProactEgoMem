from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from pydantic import ValidationError

from tests.support import install_common_stubs


install_common_stubs()

from proact.evaluation.config import (
    EvaluationConfig,
    FrameSamplingConfig,
    MessageHistoryConfig,
    MetricConfig,
    ParserConfig,
    PromptConfig,
    SilenceConfig,
    TaskPolicyConfig,
)
from proact.evaluation.io.serialization import write_json
from proact.evaluation.runtime.prompts import PromptBuilder
from proact.evaluation.runtime.parser import ConfiguredResponseParser
from proact.evaluation.runtime.records import ParsedModelOutput
from proact.evaluation.runtime.sampling import DefaultVideoSampler
from proact.evaluation.runtime.state_tracker import ToolStateTracker
from proact.evaluation.runtime.tools import build_default_tool_registry
from proact.evaluation.runtime.vidllms.base import build_interleaved_video_window_text
from proact.models.llm.base import ImageB64Part, TextPart


class EvaluationComponentTests(unittest.TestCase):
    def test_parser_normalizes_silence_and_tool_calls(self) -> None:
        parser = ConfiguredResponseParser(
            ParserConfig(
                allow_root_list=False,
                allow_root_tool_object=False,
                single_tool_keys=["tool"],
                multi_tool_keys=["tool_calls"],
                tool_name_key="name",
                tool_args_key="args",
                timestamp_keys=["timestamp_sec"],
                silence=SilenceConfig(raw_text_values=["NO_ACTION"], empty_list_means_no_action=True),
            ),
            allowed_tool_names={"start_reminder", "stop_reminder"},
        )

        silent = parser.parse("NO_ACTION")
        parsed = parser.parse(
            '{"tool": {"name": "start_reminder", "args": {"object": "map"}, "timestamp_sec": 12.5}}'
        )
        bad = parser.parse("not json")

        self.assertEqual(silent.status, "NO_ACTION")
        self.assertEqual(parsed.status, "VALID_TOOL_CALL")
        self.assertEqual(parsed.tool_calls[0].name, "start_reminder")
        self.assertEqual(parsed.tool_calls[0].timestamp_sec, 12.5)
        parsed_stop = parser.parse(
            '{"tool": {"name": "stop_reminder", "args": {"id": "reminder-1"}}}'
        )
        self.assertEqual(parsed_stop.tool_calls[0].name, "stop_reminder")
        self.assertEqual(parsed_stop.tool_calls[0].args["id"], "reminder-1")
        self.assertEqual(bad.status, "UNPARSEABLE_OUTPUT")

    def test_parser_can_extract_first_json_from_mixed_text_when_enabled(self) -> None:
        parser = ConfiguredResponseParser(
            ParserConfig(
                allow_first_json_match=True,
                allow_root_list=False,
                allow_root_tool_object=False,
                single_tool_keys=["tool"],
                multi_tool_keys=["tool_calls"],
                tool_name_key="name",
                tool_args_key="args",
                timestamp_keys=[],
            ),
            allowed_tool_names={"start_reminder", "stop_reminder"},
        )

        parsed = parser.parse(
            'Reasoning first.\nStill explaining.\n{"tool": {"name": "start_reminder", "args": {"name": "cassette", "location": "workbench"}}}'
        )

        self.assertEqual(parsed.status, "VALID_TOOL_CALL")
        self.assertEqual(parsed.tool_calls[0].name, "start_reminder")
        self.assertEqual(parsed.tool_calls[0].args["name"], "cassette")

    def test_parser_can_extract_last_json_from_mixed_text_when_enabled(self) -> None:
        parser = ConfiguredResponseParser(
            ParserConfig(
                allow_last_json_match=True,
                allow_root_list=False,
                allow_root_tool_object=False,
                single_tool_keys=["tool"],
                multi_tool_keys=["tool_calls"],
                tool_name_key="name",
                tool_args_key="args",
                timestamp_keys=[],
            ),
            allowed_tool_names={"start_reminder", "stop_reminder"},
        )

        parsed = parser.parse(
            'First thought {"decision": "no_action"} then final answer {"tool": {"name": "start_reminder", "args": {"name": "cassette", "location": "workbench"}}}'
        )

        self.assertEqual(parsed.status, "VALID_TOOL_CALL")
        self.assertEqual(parsed.tool_calls[0].name, "start_reminder")
        self.assertEqual(parsed.tool_calls[0].args["location"], "workbench")

    def test_parser_config_rejects_first_and_last_json_match_together(self) -> None:
        with self.assertRaises(ValueError):
            ParserConfig(
                allow_first_json_match=True,
                allow_last_json_match=True,
            ).validate_config()

    def test_native_streaming_config_requires_adapter_and_frame_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.txt"
            prompt_path.write_text("Prompt", encoding="utf-8")

            with self.assertRaises(ValueError):
                EvaluationConfig(
                    dataset={
                        "dataset": "ProactBenchGrouped",
                        "root": str(tmp_path),
                        "split": "val",
                        "dataset_params": {
                            "version": "v1",
                            "task_json": "task.json",
                            "ego4d_root": str(tmp_path),
                        },
                    },
                    context="long-video",
                    execution_backend={"mode": "native-streaming"},
                    task_policy={
                        "task_name": "object_tracking",
                        "expected_interval_types": ["reminder"],
                        "expected_start_tools": ["start_reminder"],
                        "expected_end_tools": ["stop_reminder"],
                        "allowed_tool_names": ["start_reminder", "stop_reminder"],
                        "ignored_tool_names": [],
                    },
                    prompt={"prompt_path": str(prompt_path)},
                    parser={},
                    metrics={
                        "proactive_start_tolerances_sec": [1.0],
                        "iou_thresholds": [0.5],
                    },
                    frame_sampling={},
                    output={"output_dir": str(tmp_path / "outputs")},
                    seed=1,
                ).validate_config()

    def test_native_streaming_config_allows_model_omission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.txt"
            prompt_path.write_text("Prompt", encoding="utf-8")

            config = EvaluationConfig(
                dataset={
                    "dataset": "ProactBenchGrouped",
                    "root": str(tmp_path),
                    "split": "val",
                    "dataset_params": {
                        "version": "v1",
                        "task_json": "task.json",
                        "ego4d_root": str(tmp_path),
                    },
                },
                context="long-video",
                execution_backend={
                    "mode": "native-streaming",
                    "adapter": "mmduet",
                    "frame_fps": 2.0,
                },
                task_policy={
                    "task_name": "object_tracking",
                    "expected_interval_types": ["reminder"],
                    "expected_start_tools": ["start_reminder"],
                    "expected_end_tools": ["stop_reminder"],
                    "allowed_tool_names": ["start_reminder", "stop_reminder"],
                    "ignored_tool_names": [],
                },
                prompt={"prompt_path": str(prompt_path)},
                parser={},
                metrics={
                    "proactive_start_tolerances_sec": [1.0],
                    "iou_thresholds": [0.5],
                },
                frame_sampling={},
                output={"output_dir": str(tmp_path / "outputs")},
                seed=1,
            ).validate_config()

            self.assertEqual(config.execution_backend.mode, "native-streaming")
            self.assertIsNone(config.model)
            self.assertTrue(config.execution_backend.inject_tool_state_updates)

    def test_video_llm_config_requires_adapter_and_input_mode_but_not_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.txt"
            prompt_path.write_text("Prompt", encoding="utf-8")

            common = {
                "dataset": {
                    "dataset": "ProactBenchGrouped",
                    "root": str(tmp_path),
                    "split": "val",
                    "dataset_params": {
                        "version": "v1",
                        "task_json": "task.json",
                        "ego4d_root": str(tmp_path),
                    },
                },
                "context": "long-video",
                "task_policy": {
                    "task_name": "object_tracking",
                    "expected_interval_types": ["reminder"],
                    "expected_start_tools": ["start_reminder"],
                    "expected_end_tools": ["stop_reminder"],
                    "allowed_tool_names": ["start_reminder", "stop_reminder"],
                    "ignored_tool_names": [],
                },
                "prompt": {"prompt_path": str(prompt_path)},
                "parser": {},
                "metrics": {
                    "proactive_start_tolerances_sec": [1.0],
                    "iou_thresholds": [0.5],
                },
                "frame_sampling": {},
                "output": {"output_dir": str(tmp_path / "outputs")},
                "seed": 1,
            }

            with self.assertRaises(ValueError):
                EvaluationConfig(
                    **common,
                    execution_backend={"mode": "video-llm"},
                    input_mode={
                        "mode": "sliding-window",
                        "fps": 1.0,
                        "window_size_frames": 4,
                        "stride_frames": 1,
                    },
                ).validate_config()

            with self.assertRaises(ValueError):
                EvaluationConfig(
                    **common,
                    execution_backend={"mode": "video-llm", "adapter": "mmduet"},
                    input_mode={
                        "mode": "sliding-window",
                        "fps": 1.0,
                        "window_size_frames": 4,
                        "stride_frames": 1,
                    },
                ).validate_config()

            config = EvaluationConfig(
                **common,
                execution_backend={"mode": "video-llm", "adapter": "minicpm-o45"},
                input_mode={
                    "mode": "sliding-window",
                    "fps": 1.0,
                    "window_size_frames": 4,
                    "stride_frames": 1,
                },
            ).validate_config()

            self.assertEqual(config.execution_backend.mode, "video-llm")
            self.assertIsNone(config.model)

    def test_tracker_handles_overlay_lifecycle_and_invalid_stop(self) -> None:
        tracker = ToolStateTracker(build_default_tool_registry())

        step0 = tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[
                    parser_call("start_reminder", {"object": "map"}),
                    parser_call("start_reminder", {"object": "keys"}),
                ],
            ),
            step_index=0,
            step_time_sec=5.0,
            sampled_frame_times=[5.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )
        step1 = tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[parser_call("stop_reminder", {"id": "unknown-overlay"})],
            ),
            step_index=1,
            step_time_sec=9.0,
            sampled_frame_times=[9.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )
        step2 = tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[parser_call("stop_reminder", {"id": "reminder-1"})],
            ),
            step_index=2,
            step_time_sec=12.0,
            sampled_frame_times=[12.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        intervals = tracker.finalize()

        self.assertEqual(len(step0.state_after["active_intervals"]), 2)
        self.assertEqual(len(step1.invalid_actions), 1)
        self.assertEqual(intervals[0].instance_id, "reminder-1")
        self.assertFalse(intervals[0].is_open)
        self.assertEqual(intervals[0].predicted_end_time, 12.0)
        self.assertTrue(intervals[1].is_open)
        self.assertEqual(intervals[1].args["object"], "keys")

    def test_active_only_prompt_state_uses_dense_templates(self) -> None:
        tracker = ToolStateTracker(build_default_tool_registry())
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="raw model output should not be included",
                tool_calls=[parser_call("start_reminder", {"object": "map"})],
            ),
            step_index=0,
            step_time_sec=5.0,
            sampled_frame_times=[5.0],
            request_messages=[{"role": "user", "content": "previous prompt"}],
            raw_model_output={"response_text": "previous raw output"},
            allow_implicit_timestamps=True,
        )
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[parser_call("stop_reminder", {"id": "reminder-1"})],
            ),
            step_index=1,
            step_time_sec=9.0,
            sampled_frame_times=[9.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        active_state = tracker.prompt_state(
            state_format="active-only",
            tool_state_templates={
                "start_reminder": {"message": "tracking {object}", "forwarded_args": ["object"]},
                "stop_reminder": {"message": "reminder off right now", "forwarded_args": []},
            },
        )

        self.assertEqual(active_state["state_messages"], ["reminder off right now"])

    def test_sampler_resizes_frames_without_changing_timestamps(self) -> None:
        class DummyFrame:
            def __init__(self):
                self.time = 1.5
                self.chosen_for_time = 1.5
                self.array = "original"

        fake_media = ModuleType("proact.utils.media")
        fake_media.resize_frame = lambda array, width, height: f"resized:{array}:{width}x{height}"

        sampler = DefaultVideoSampler(
            FrameSamplingConfig(
                resize_frames=True,
                resize_width=224,
                resize_height=224,
            )
        )
        frames = [DummyFrame()]

        with mock.patch.dict(sys.modules, {"proact.utils.media": fake_media}):
            resized = sampler._resize_frames_if_needed(frames)

        self.assertEqual(resized[0].array, "resized:original:224x224")
        self.assertEqual(resized[0].time, 1.5)
        self.assertEqual(resized[0].chosen_for_time, 1.5)
        self.assertEqual(
            sampler._frame_resize_metadata(),
            {"enabled": True, "width": 224, "height": 224},
        )

    def test_active_only_prompt_state_logs_missing_forwarded_args(self) -> None:
        tracker = ToolStateTracker(build_default_tool_registry())
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[parser_call("start_reminder", {})],
            ),
            step_index=0,
            step_time_sec=3.0,
            sampled_frame_times=[3.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        with mock.patch("builtins.print") as mocked_print:
            state = tracker.prompt_state(
                state_format="active-only",
                tool_state_templates={
                    "start_reminder": {"message": "tracking {object}", "forwarded_args": ["object"]},
                    "stop_reminder": {"message": "overlay off", "forwarded_args": []},
                },
            )

        self.assertEqual(state["state_messages"], ["tracking "])
        mocked_print.assert_called_once()
        self.assertIn("Missing forwarded arg 'object'", mocked_print.call_args.args[0])

    def test_active_only_prompt_state_can_reference_generated_reminder_id(self) -> None:
        tracker = ToolStateTracker(build_default_tool_registry())
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="VALID_TOOL_CALL",
                raw_text="",
                tool_calls=[parser_call("start_reminder", {"name": "keys", "location": "desk"})],
            ),
            step_index=0,
            step_time_sec=3.0,
            sampled_frame_times=[3.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        state = tracker.prompt_state(
            state_format="active-only",
            tool_state_templates={
                "start_reminder": {
                    "message": "Active track [{id}] since {start_time_sec:.3f}s: {name} at {location}",
                    "forwarded_args": ["name", "location"],
                },
                "stop_reminder": {"message": "There are currently no active tracks", "forwarded_args": []},
            },
        )

        self.assertEqual(
            state["state_messages"],
            ["Active track [reminder-1] since 3.000s: keys at desk"],
        )

    def test_message_history_tracks_bounded_message_outputs(self) -> None:
        tracker = ToolStateTracker(
            build_default_tool_registry(),
            message_history_config=MessageHistoryConfig(
                enabled=True,
                message_key="message",
                max_messages=2,
            ),
        )

        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="INVALID_TOOL_CALL",
                raw_text="",
                parsed_payload={"act": True, "message": "first note"},
                issues=["missing tool name"],
            ),
            step_index=0,
            step_time_sec=1.0,
            sampled_frame_times=[1.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )
        step1 = tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="INVALID_TOOL_CALL",
                raw_text="",
                parsed_payload={"act": True, "message": "second note"},
                issues=["missing tool name"],
            ),
            step_index=1,
            step_time_sec=2.5,
            sampled_frame_times=[2.5],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="INVALID_TOOL_CALL",
                raw_text="",
                parsed_payload={"act": True, "message": "third note"},
                issues=["missing tool name"],
            ),
            step_index=2,
            step_time_sec=4.0,
            sampled_frame_times=[4.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        state = tracker.prompt_state(
            state_format="active-only",
            tool_state_templates={
                "start_recording": {"message": "recording {name}", "forwarded_args": ["name"]},
                "stop_recording": {"message": "not recording", "forwarded_args": []},
            },
            include_tool_state=False,
        )

        self.assertEqual(step1.state_after["recent_messages"][1]["message"], "second note")
        self.assertEqual(
            state["recent_messages"],
            [
                {"timestamp_sec": 2.5, "step_index": 1, "message": "second note"},
                {"timestamp_sec": 4.0, "step_index": 2, "message": "third note"},
            ],
        )
        self.assertNotIn("state_messages", state)

    def test_message_history_skips_silence_by_default(self) -> None:
        tracker = ToolStateTracker(
            build_default_tool_registry(),
            message_history_config=MessageHistoryConfig(enabled=True),
        )
        tracker.consume_step(
            parsed_output=ParsedModelOutput(
                status="NO_ACTION",
                raw_text="",
                parsed_payload={"act": False, "message": "silent thought"},
            ),
            step_index=0,
            step_time_sec=1.0,
            sampled_frame_times=[1.0],
            request_messages=[],
            raw_model_output={},
            allow_implicit_timestamps=True,
        )

        state = tracker.prompt_state(
            state_format="active-only",
            tool_state_templates={},
            include_tool_state=False,
        )

        self.assertEqual(state["recent_messages"], [])

    def test_prompt_config_rejects_legacy_prompt_role(self) -> None:
        with self.assertRaises(ValidationError):
            PromptConfig(prompt_path="/tmp/prompt.txt", prompt_role="system")

    def test_prompt_builder_orders_prompt_then_state_then_images(self) -> None:
        class DummyFrame:
            def __init__(self, time: float):
                self.time = time
                self.chosen_for_time = time
                self.array = ImageB64Part(b64="AA==", mime_type="image/png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Loaded prompt body", encoding="utf-8")

            builder = PromptBuilder(
                PromptConfig(
                    prompt_path=str(prompt_path),
                    response_format_prompt="Return strict JSON.",
                    state_format="active-only",
                )
            )
            messages = builder.build_messages(
                frames=[DummyFrame(41.033), DummyFrame(42.125)],
                tool_state={"state_messages": ["There are currently no active tracks"]},
                step_index=0,
                timestamp_sec=42.125,
                evaluation_context="short-video",
                input_mode="sliding-window",
                input_video_path="/tmp/video.mp4",
                metadata={},
            )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        content = messages[0].content
        self.assertEqual(len(content), 7)
        self.assertIsInstance(content[0], TextPart)
        self.assertEqual(content[0].text, "Loaded prompt body\n\nReturn strict JSON.")
        self.assertIsInstance(content[1], TextPart)
        self.assertTrue(content[1].text.startswith("Current assistant context"))
        self.assertIn('"current_time_sec": 42.125', content[1].text)
        self.assertIn('"state_messages"', content[1].text)
        self.assertIn("There are currently no active tracks", content[1].text)
        self.assertIsInstance(content[2], TextPart)
        self.assertEqual(content[2].text, "The window of video frames")
        self.assertIsInstance(content[3], ImageB64Part)
        self.assertIsNone(content[3].alt)
        self.assertIsInstance(content[4], TextPart)
        self.assertEqual(content[4].text, "Frame #1 at 41.033 seconds")
        self.assertIsInstance(content[5], ImageB64Part)
        self.assertIsNone(content[5].alt)
        self.assertIsInstance(content[6], TextPart)
        self.assertEqual(content[6].text, "Frame #2 at 42.125 seconds")

    def test_prompt_builder_passes_prompt_language_through_verbatim(self) -> None:
        # There is no alias/translation layer any more: the repo-wide vocabulary
        # is `reminder`, and prompt text must reach the model unchanged (a model
        # message mentioning e.g. "overlay" must never be rewritten).
        class DummyFrame:
            def __init__(self, time: float):
                self.time = time
                self.chosen_for_time = time
                self.array = ImageB64Part(b64="AA==", mime_type="image/png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text(
                "Use start_reminder and stop_reminder. Several reminders may be "
                "active. Use the matching reminder-id.",
                encoding="utf-8",
            )

            builder = PromptBuilder(
                PromptConfig(
                    prompt_path=str(prompt_path),
                    state_format="active-only",
                )
            )
            messages = builder.build_messages(
                frames=[DummyFrame(1.0)],
                tool_state={
                    "active_tools": [
                        {
                            "instance_id": "reminder-1",
                            "tool_name": "start_reminder",
                            "started_at_sec": 1.0,
                            "arguments": {"name": "lamp", "state": "on"},
                        }
                    ]
                },
                step_index=0,
                timestamp_sec=1.0,
                evaluation_context="short-video",
                input_mode="sliding-window",
                input_video_path="/tmp/video.mp4",
                metadata={},
            )

        rendered = messages[0].to_string(include_role=False, include_alt=True)
        self.assertIn("start_reminder", rendered)
        self.assertIn("stop_reminder", rendered)
        self.assertIn("reminders may be active", rendered)
        self.assertIn("reminder-id", rendered)
        self.assertIn('"instance_id": "reminder-1"', rendered)

    def test_video_llm_trace_prompt_uses_interleaved_image_timestamp_shape(self) -> None:
        rendered = build_interleaved_video_window_text(
            prompt_text="Loaded prompt body",
            state_text="Current assistant context\n\n{}",
            frame_timestamps=[0.0, 1.25],
            image_placeholder="[IMAGE]",
        )

        self.assertEqual(
            rendered,
            "Loaded prompt body\n\n"
            "Current assistant context\n\n"
            "{}\n\n"
            "The window of video frames\n"
            "[IMAGE]\n"
            "Frame #1 at 0.000 seconds\n"
            "[IMAGE]\n"
            "Frame #2 at 1.250 seconds",
        )

    def test_prompt_builder_image_cache_key_depends_on_frame_content(self) -> None:
        import numpy as np

        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        first_key = PromptBuilder._image_cache_key(frame)
        frame[:, :, :] = 255
        second_key = PromptBuilder._image_cache_key(frame)

        self.assertNotEqual(first_key, second_key)

    def test_prompt_builder_can_omit_state_text_from_model_request(self) -> None:
        class DummyFrame:
            def __init__(self, time: float):
                self.time = time
                self.chosen_for_time = time
                self.array = ImageB64Part(b64="AA==", mime_type="image/png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Loaded prompt body", encoding="utf-8")

            builder = PromptBuilder(
                PromptConfig(
                    prompt_path=str(prompt_path),
                    include_state=False,
                    state_format="active-only",
                )
            )
            messages = builder.build_messages(
                frames=[DummyFrame(42.125)],
                tool_state={"state_messages": ["This state should stay internal"]},
                step_index=0,
                timestamp_sec=42.125,
                evaluation_context="short-video",
                input_mode="sliding-window",
                input_video_path="/tmp/video.mp4",
                metadata={},
            )

        content = messages[0].content
        self.assertEqual(len(content), 4)
        self.assertEqual(content[0].text, "Loaded prompt body")
        self.assertEqual(content[1].text, "The window of video frames")
        self.assertIsInstance(content[2], ImageB64Part)
        self.assertEqual(content[3].text, "Frame #1 at 42.125 seconds")
        self.assertNotIn(
            "Current assistant context",
            messages[0].to_string(include_role=False, include_alt=True),
        )
        self.assertNotIn(
            "This state should stay internal",
            messages[0].to_string(include_role=False, include_alt=True),
        )

    def test_prompt_builder_includes_message_history_when_tool_state_is_disabled(self) -> None:
        class DummyFrame:
            def __init__(self, time: float):
                self.time = time
                self.chosen_for_time = time
                self.array = ImageB64Part(b64="AA==", mime_type="image/png")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Loaded prompt body", encoding="utf-8")

            builder = PromptBuilder(
                PromptConfig(
                    prompt_path=str(prompt_path),
                    include_state=False,
                    state_format="active-only",
                    message_history=MessageHistoryConfig(enabled=True, max_messages=10),
                )
            )
            messages = builder.build_messages(
                frames=[DummyFrame(42.125)],
                tool_state={
                    "recent_messages": [
                        {"timestamp_sec": 4.5, "step_index": 2, "message": "record the sign"}
                    ]
                },
                step_index=3,
                timestamp_sec=42.125,
                evaluation_context="short-video",
                input_mode="sliding-window",
                input_video_path="/tmp/video.mp4",
                metadata={},
            )

        rendered = messages[0].to_string(include_role=False, include_alt=True)
        self.assertIn("Current assistant context", rendered)
        self.assertIn('"recent_messages"', rendered)
        self.assertNotIn('"tool_state"', rendered)
        self.assertIn("record the sign", rendered)
        self.assertIn('"timestamp_sec": 4.5', rendered)

    def test_write_json_serializes_provider_raw_objects(self) -> None:
        class RawOutput:
            def to_dict(self):
                return {"nested": {"value": "leaf-value"}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "execution_traces.json"
            write_json(path, [{"raw_model_output": {"response_raw": RawOutput()}}])
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload[0]["raw_model_output"]["response_raw"]["nested"]["value"], "leaf-value")

    def test_write_json_does_not_truncate_existing_file_when_conversion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "execution_traces.json"
            path.write_text("[\"previous\"]\n", encoding="utf-8")
            circular = []
            circular.append(circular)

            with self.assertRaises(RecursionError):
                write_json(path, circular)

            self.assertEqual(path.read_text(encoding="utf-8"), "[\"previous\"]\n")


def parser_call(name: str, args: dict[str, object]):
    from proact.evaluation.runtime.records import NormalizedToolCall

    return NormalizedToolCall(name=name, args=args)


if __name__ == "__main__":
    unittest.main()
