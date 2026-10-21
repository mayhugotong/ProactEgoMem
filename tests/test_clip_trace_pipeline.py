from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.support import install_common_stubs

install_common_stubs()

import numpy as np

from proact.data.datasets.proact_bench import ProactBenchClips
from proact.evaluation.clip_artifacts import (
    build_proactive_memory,
    render_proactive_memory,
)
from proact.evaluation.clip_pipeline import ClipTraceRunner, _shard_for_clip
from proact.evaluation.config import (
    EvaluationConfig,
    ParserConfig,
    PromptContextConfig,
    load_evaluation_config,
)
from proact.evaluation.runtime.execution import StatelessExecutionBackend
from proact.evaluation.runtime.parser import ConfiguredResponseParser
from proact.evaluation.runtime.prompts import PromptBuilder
from proact.evaluation.runtime.records import (
    INVALID_TOOL_CALL,
    NO_ACTION,
    UNPARSEABLE_OUTPUT,
    VALID_OUTPUT,
)
from proact.evaluation.runtime.sampling import DefaultVideoSampler, SampleBatch
from proact.evaluation.runtime.streaming import NativeStreamingExecutionBackend
from proact.evaluation.runtime.streaming.adapters import BufferedStreamingSession, build_streaming_adapter
from proact.evaluation.runtime.streaming.base import StreamingOutputEvent
from proact.evaluation.runtime.state_tracker import ToolStateTracker
from proact.evaluation.runtime.tools import build_default_tool_registry
from proact.evaluation.runtime.vidllms import VideoLLMExecutionBackend
from proact.evaluation.runtime.vidllms.adapters import build_vidllm_adapter
from proact.evaluation.runtime.vidllms.base import VideoLLMResponse


class DummyFrame:
    def __init__(self, timestamp: float, array=None):
        self.time = timestamp
        self.chosen_for_time = timestamp
        self.array = np.zeros((2, 2, 3), dtype=np.uint8) if array is None else array


class StubSampler:
    def generate_batches(self, video_path, input_mode):
        frame = DummyFrame(1.0)
        return [SampleBatch(0, 1.0, [frame], {"sampled_frame_fingerprints": ["frame-1"]})], {
            "sampling_mode": input_mode.mode,
            "input_video_path": video_path,
            "sampled_frame_count_total": 1,
            "sampled_frame_times": [1.0],
            "sampled_frame_fingerprints": ["frame-1"],
        }

    def generate_stream_frames(self, video_path, fps):
        frame = DummyFrame(1.0)
        return [frame], {
            "sampling_mode": "native-streaming",
            "input_video_path": video_path,
            "sampled_frame_count_total": 1,
            "sampled_frame_times": [1.0],
            "sampled_frame_fingerprints": ["frame-1"],
        }


class StubModelRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.prompt_builder = None

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return {
            "request_messages": [{"role": "user", "content": "prompt\n[IMAGE]"}],
            "response_ok": True,
            "response_text": response,
            "response_reasoning": None,
            "response_finish_reason": "stop",
            "response_error": None,
            "response_error_type": None,
            "response_status_code": 200,
            "response_raw": {"text": response},
        }


def write_fixture(base: Path, *, clip_count: int = 2):
    prompt = base / "prompt.txt"
    prompt.write_text("Observe the clip and respond using the required JSON schema.", encoding="utf-8")
    clips = {}
    videos = {"video-a": {"video_uid": "video-a", "samples": {}}}
    clip_root = base / "clips"
    clip_root.mkdir()
    for index in range(clip_count):
        uid = f"clip-{index}"
        clip_path = clip_root / f"{uid}.mp4"
        clip_path.write_bytes(b"video")
        clips[uid] = {
            "clip_uid": uid,
            "video_uid": "video-a",
            "clip_path": str(clip_path),
            "clip_duration_sec": 4.0 + index,
            "sample_entry_ids": ["stale-id"],
        }
    annotation = base / "annotations.json"
    annotation.write_text(json.dumps({
        "schema_version": "test-v1",
        "dataset_version": "test",
        "task": "state_tracking",
        "videos": videos,
        "clip_index": clips,
    }), encoding="utf-8")
    return prompt, annotation, clip_root


def make_config(base: Path, prompt: Path, annotation: Path, *, proactive_level="intervention"):
    return EvaluationConfig(
        dataset={
            "dataset": "ProactBenchClips",
            "root": str(base),
            "dataset_params": {"task_json": str(annotation)},
        },
        proactive_level=proactive_level,
        input_mode={
            "mode": "sliding-window",
            "fps": 1.0,
            "window_size_frames": 1,
            "stride_frames": 1,
        },
        task_policy={
            "task_name": "state_tracking",
            "expected_interval_types": ["reminder"],
            "expected_start_tools": ["start_reminder"],
            "expected_end_tools": ["stop_reminder"],
            "allowed_tool_names": ["start_reminder", "stop_reminder"],
        },
        prompt={
            "prompt_path": str(prompt),
            "response_format_prompt": (
                'Return exactly {"message": null|string, "tool_calls": '
                '[{"name": string, "arguments": object}]}.'
            ),
        },
        prompt_context={
            "include_active_tools": True,
            "recent_tool_calls": {"mode": "all", "source": "attempted"},
            "recent_messages": {"mode": "all"},
        },
        parser={},
        frame_sampling={},
        model={"llm_config_path": str(prompt), "profile": "unused"},
        output={"output_dir": str(base / "run")},
    ).validate_config()


class EvaluationEnvironmentTests(unittest.TestCase):
    def test_unset_environment_variable_fails_before_path_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "output:\n  output_dir: ${PROACT_TEST_MISSING_OUTPUT}/outputs/run\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment.pop("PROACT_TEST_MISSING_OUTPUT", None)
            with mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "unset environment variables: PROACT_TEST_MISSING_OUTPUT",
                ):
                    load_evaluation_config(config_path)


    def test_raw_notification_text_is_rejected_outside_native_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, _clip_root = write_fixture(base, clip_count=1)
            config = make_config(
                base,
                prompt,
                annotation,
                proactive_level="notification",
            )
            config.parser.allow_notification_raw_text = True
            with self.assertRaisesRegex(
                ValueError,
                "only valid for notification native-streaming clip traces",
            ):
                config.validate_config()


class ProgressLoggingTests(unittest.TestCase):
    def test_eta_uses_executed_clip_average(self):
        with mock.patch(
            "proact.evaluation.clip_pipeline.time.perf_counter",
            return_value=20.0,
        ):
            with self.assertLogs("proact.evaluation.clip_pipeline", level="INFO") as logs:
                ClipTraceRunner._log_progress(
                    position=2,
                    total=4,
                    completed=1,
                    resumed=1,
                    failed=0,
                    started_at=0.0,
                    executed_clip_count=1,
                    execution_elapsed_sec=5.0,
                    current_clip_uid="clip-1",
                )
        message = logs.output[0]
        self.assertIn("elapsed=00:00:20", message)
        self.assertIn("avg_executed=00:00:05", message)
        self.assertIn("ETA=00:00:10", message)



class CanonicalDatasetTests(unittest.TestCase):
    def test_clip_index_is_the_execution_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _prompt, annotation, _clip_root = write_fixture(base)
            dataset = ProactBenchClips(root=str(base), task_json=str(annotation))
            self.assertEqual([dataset[i]["clip_uid"] for i in range(len(dataset))], ["clip-0", "clip-1"])
            self.assertNotIn("sample_entry_ids", dataset[0])


class UnifiedParserTests(unittest.TestCase):
    def parser(self, level="intervention", *, allow_notification_raw_text=False):
        return ConfiguredResponseParser(
            ParserConfig(
                unified_schema=True,
                allow_notification_raw_text=allow_notification_raw_text,
            ),
            build_default_tool_registry().names(),
            proactive_level=level,
        )

    def test_all_intervention_output_combinations(self):
        cases = [
            ('{"message": null, "tool_calls": []}', NO_ACTION),
            ('{"message": "hello", "tool_calls": []}', VALID_OUTPUT),
            ('{"message": null, "tool_calls": [{"name": "start_reminder", "arguments": {}}]}', VALID_OUTPUT),
            ('{"message": "done", "tool_calls": [{"name": "start_reminder", "arguments": {}}]}', VALID_OUTPUT),
        ]
        for payload, status in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.parser().parse(payload).status, status)

    def test_notification_uses_message_only_json(self):
        parser = self.parser("notification")
        cases = [
            ('{"message": "You just left the light on."}', VALID_OUTPUT),
            ('{"message": null}', NO_ACTION),
            ('{"message": "   "}', NO_ACTION),
            ('{"message": "hello", "extra": 1}', INVALID_TOOL_CALL),
            ('{"act": true, "message": "hello"}', INVALID_TOOL_CALL),
            ('{"message": "hello", "tool_calls": []}', INVALID_TOOL_CALL),
            ('plain notification', UNPARSEABLE_OUTPUT),
        ]
        for payload, status in cases:
            with self.subTest(payload=payload):
                parsed = parser.parse(payload)
                self.assertEqual(parsed.status, status)
                self.assertEqual(parsed.tool_calls, [])

    def test_notification_raw_text_is_streaming_compatibility_only(self):
        parser = self.parser("notification", allow_notification_raw_text=True)
        sentence = parser.parse("I can see a phone number on the sign.")
        self.assertEqual(sentence.status, VALID_OUTPUT)
        self.assertEqual(sentence.message, "I can see a phone number on the sign.")
        self.assertEqual(
            sentence.parsed_payload,
            {"message": "I can see a phone number on the sign."},
        )
        self.assertEqual(parser.parse("   ").status, NO_ACTION)
        self.assertEqual(parser.parse('{"message": "JSON is also valid."}').status, VALID_OUTPUT)
        self.assertEqual(parser.parse('{"message":').status, UNPARSEABLE_OUTPUT)
        self.assertEqual(parser.parse('```json\n{"message":\n```').status, UNPARSEABLE_OUTPUT)

    def test_intervention_uses_last_json_from_noisy_response(self):
        parsed = self.parser().parse(
            'Draft answer:\n```json\n{"message": null, "tool_calls": []}\n```\n'
            'Final answer: {"message": "tracking", "tool_calls": '
            '[{"name": "start_reminder", "arguments": {"name": "kettle"}}]}'
        )

        self.assertEqual(parsed.status, VALID_OUTPUT)
        self.assertEqual(parsed.message, "tracking")
        self.assertEqual(parsed.tool_calls[0].name, "start_reminder")
        self.assertEqual(parsed.tool_calls[0].args["name"], "kettle")

    def test_notification_uses_last_json_from_noisy_response(self):
        parsed = self.parser("notification").parse(
            'Earlier candidate {"message": "first"}\nFinal answer: {"message": "second"}'
        )

        self.assertEqual(parsed.status, VALID_OUTPUT)
        self.assertEqual(parsed.message, "second")

    def test_intervention_schema_remains_exact(self):
        malformed = self.parser().parse('{"message": null, "tool_calls": [], "extra": 1}')
        self.assertEqual(malformed.status, INVALID_TOOL_CALL)


class CollatedIntervalHistoryTests(unittest.TestCase):
    def test_compact_state_collates_completed_intervals_and_keeps_active_separate(self):
        context = PromptContextConfig(
            include_active_tools=True,
            completed_tool_intervals={"mode": "last_n", "n": 1},
        )
        parser = ConfiguredResponseParser(
            ParserConfig(unified_schema=True),
            build_default_tool_registry().names(),
        )
        tracker = ToolStateTracker(
            build_default_tool_registry(),
            prompt_context_config=context,
        )

        def consume(step_index, timestamp, payload):
            tracker.consume_step(
                parsed_output=parser.parse(payload),
                step_index=step_index,
                step_time_sec=timestamp,
                sampled_frame_times=[timestamp],
                request_messages=[],
                raw_model_output={"response_text": payload},
                allow_implicit_timestamps=True,
            )

        consume(
            0,
            2.0,
            '{"message": null, "tool_calls": [{"name": "start_reminder", "arguments": {"name": "faucet", "state": "on"}}]}',
        )
        consume(
            1,
            6.0,
            '{"message": null, "tool_calls": [{"name": "stop_reminder", "arguments": {"id": "reminder-1"}}]}',
        )
        consume(
            2,
            8.0,
            '{"message": null, "tool_calls": [{"name": "start_reminder", "arguments": {"name": "oven", "state": "on"}}]}',
        )

        state = tracker.prompt_state(state_format="compact", include_tool_state=True)
        self.assertEqual(state["active_tools"], [{
            "instance_id": "reminder-2",
            "tool_name": "start_reminder",
            "started_at_sec": 8.0,
            "arguments": {"name": "oven", "state": "on"},
        }])
        self.assertEqual(state["completed_tool_intervals"], [{
            "instance_id": "reminder-1",
            "start_tool_name": "start_reminder",
            "end_tool_name": "stop_reminder",
            "started_at_sec": 2.0,
            "ended_at_sec": 6.0,
            "arguments": {"name": "faucet", "state": "on"},
        }])
        self.assertNotIn("recent_tool_calls", state)



class NotificationBackendContractTests(unittest.TestCase):
    @staticmethod
    def assert_notification_trace(trace, expected_message):
        assert len(trace.output_events) == 1
        event = trace.output_events[0]
        assert event.parsed_message == expected_message
        assert event.parsed_tool_calls == []
        assert event.requested_tool_calls == []
        assert event.executed_tool_calls == []
        assert event.state_transitions == []
        assert event.invalid_actions == []
        assert trace.predicted_intervals == []
        assert trace.final_state["active_intervals"] == []

    def test_stateless_notification_trace_normalizes_without_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(
                base,
                prompt,
                annotation,
                proactive_level="notification",
            )
            model = StubModelRunner(['{"message": "You just left the faucet running."}'])
            backend = StatelessExecutionBackend(
                config=config,
                parser=ConfiguredResponseParser(
                    config.parser,
                    build_default_tool_registry().names(),
                    proactive_level="notification",
                ),
                tool_registry=build_default_tool_registry(),
                model_runner=model,
                video_sampler=StubSampler(),
            )
            trace = backend.execute_trace(
                trace_id="clip-0",
                source_unit_id="clip-0",
                input_video_path=str(clip_root / "clip-0.mp4"),
                trace_metadata={},
            )
            self.assert_notification_trace(trace, "You just left the faucet running.")

    def test_video_llm_notification_trace_normalizes_without_tools(self):
        class FakeVideoAdapter:
            def complete_window(self, request):
                return VideoLLMResponse(
                    response_text='{"message": "You just put the keys on the shelf."}',
                    request_messages=[{"role": "user", "content": request.input_text}],
                )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(
                base,
                prompt,
                annotation,
                proactive_level="notification",
            )
            config.execution_backend.mode = "video-llm"
            config.execution_backend.adapter = "fake"
            builder = PromptBuilder(config.prompt, config.prompt_context)
            backend = VideoLLMExecutionBackend(
                config=config,
                parser=ConfiguredResponseParser(
                    config.parser,
                    build_default_tool_registry().names(),
                    proactive_level="notification",
                ),
                tool_registry=build_default_tool_registry(),
                prompt_builder=builder,
                video_sampler=StubSampler(),
                adapter=FakeVideoAdapter(),
            )
            trace = backend.execute_trace(
                trace_id="clip-0",
                source_unit_id="clip-0",
                input_video_path=str(clip_root / "clip-0.mp4"),
                trace_metadata={},
            )
            self.assert_notification_trace(trace, "You just put the keys on the shelf.")

    def test_streaming_notification_trace_normalizes_raw_sentence_without_tools(self):
        class FakeStreamingSession:
            def __init__(self):
                self.emitted = False

            def reset(self):
                self.emitted = False

            def inject_text(self, *, text, timestamp_sec, metadata=None):
                pass

            def ingest_frame(self, *, frame, timestamp_sec, metadata=None):
                if self.emitted:
                    return []
                self.emitted = True
                return [
                    StreamingOutputEvent(
                        timestamp_sec=timestamp_sec,
                        response_text="I can see the Wi-Fi password on the sign.",
                        sampled_frame_times=[timestamp_sec],
                    )
                ]

            def finalize(self):
                return []

        class FakeStreamingAdapter:
            def create_session(self, **kwargs):
                return FakeStreamingSession()

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(
                base,
                prompt,
                annotation,
                proactive_level="notification",
            )
            config.execution_backend.mode = "native-streaming"
            config.execution_backend.adapter = "fake"
            config.execution_backend.frame_fps = 1.0
            config.parser.allow_notification_raw_text = True
            config.prompt_context.include_active_tools = False
            config.prompt_context.recent_tool_calls.mode = "none"
            config.prompt_context.recent_messages.mode = "last_n"
            builder = PromptBuilder(config.prompt, config.prompt_context)
            backend = NativeStreamingExecutionBackend(
                config=config,
                parser=ConfiguredResponseParser(
                    config.parser,
                    build_default_tool_registry().names(),
                    proactive_level="notification",
                ),
                tool_registry=build_default_tool_registry(),
                prompt_builder=builder,
                video_sampler=StubSampler(),
                adapter=FakeStreamingAdapter(),
            )
            trace = backend.execute_trace(
                trace_id="clip-0",
                source_unit_id="clip-0",
                input_video_path=str(clip_root / "clip-0.mp4"),
                trace_metadata={},
            )
            self.assert_notification_trace(
                trace,
                "I can see the Wi-Fi password on the sign.",
            )

    def test_notification_clip_bundle_contains_messages_but_no_tool_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(
                base,
                prompt,
                annotation,
                proactive_level="notification",
            )
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "You just left the cupboard open."}',
                ]),
                video_sampler=StubSampler(),
            ).run()
            self.assertEqual(result["completed_clip_count"], 1)

            clip_dir = (
                base / "run" / "shards" / "shard-00000-of-00001" / "clips" / "clip-0"
            )
            trace_payload = json.loads((clip_dir / "trace.json").read_text(encoding="utf-8"))
            memory = json.loads(
                (clip_dir / "proactive_memory.json").read_text(encoding="utf-8")
            )
            trace = trace_payload["trace"]
            event = trace["output_events"][0]
            self.assertEqual(event["parsed_message"], "You just left the cupboard open.")
            self.assertEqual(event["parsed_tool_calls"], [])
            self.assertEqual(event["requested_tool_calls"], [])
            self.assertEqual(event["executed_tool_calls"], [])
            self.assertEqual(event["state_transitions"], [])
            self.assertEqual(trace["predicted_intervals"], [])
            self.assertEqual(memory["collated_tool_intervals"], [])
            self.assertEqual(
                memory["non_silent_output_events"][0]["message"],
                "You just left the cupboard open.",
            )
            markdown = (clip_dir / "proactive_memory.md").read_text(encoding="utf-8")
            self.assertIn("You just left the cupboard open.", markdown)
            self.assertIn("_No executed tool intervals._", markdown)


class ProactiveMemoryTests(unittest.TestCase):
    def trace_payload(self):
        return {
            "trace_schema_version": "proact.clip-trace.v2",
            "clip": {
                "clip_uid": "clip-memory",
                "video_uid": "video-memory",
                "clip_duration_sec": 10.0,
            },
            "trace": {
                "trace_schema_version": "proact.clip-trace.v2",
                "predicted_intervals": [
                    {
                        "interval_id": "reminder-2",
                        "instance_id": "reminder-2",
                        "interval_type": "reminder",
                        "tool_start_name": "start_reminder",
                        "tool_end_name": None,
                        "predicted_start_time": 2.0,
                        "predicted_end_time": None,
                        "is_open": True,
                        "args": {"object": "oven"},
                        "start_step_index": 2,
                        "end_step_index": None,
                    },
                    {
                        "interval_id": "reminder-1",
                        "instance_id": "reminder-1",
                        "interval_type": "reminder",
                        "tool_start_name": "start_reminder",
                        "tool_end_name": "stop_reminder",
                        "predicted_start_time": 2.0,
                        "predicted_end_time": 5.0,
                        "is_open": False,
                        "args": {"object": "faucet"},
                        "start_step_index": 1,
                        "end_step_index": 3,
                    },
                ],
                "output_events": [
                    {
                        "step_index": 4,
                        "timestamp_sec": 6.0,
                        "parser_status": NO_ACTION,
                        "parser_issues": [],
                        "parser_matched_silence_rule": True,
                        "parsed_message": None,
                        "parsed_payload": {"message": None, "tool_calls": []},
                        "parsed_tool_calls": [],
                        "requested_tool_calls": [],
                        "executed_tool_calls": [],
                        "parser_invalid_entries": [],
                        "invalid_actions": [],
                        "raw_model_output": {
                            "response_text": '{"message":null,"tool_calls":[]}',
                            "response_raw": {"large": "provider envelope"},
                        },
                    },
                    {
                        "step_index": 5,
                        "timestamp_sec": 7.0,
                        "parser_status": UNPARSEABLE_OUTPUT,
                        "parser_issues": ["Response is not valid JSON."],
                        "parser_matched_silence_rule": None,
                        "parsed_message": None,
                        "parsed_payload": None,
                        "parsed_tool_calls": [],
                        "requested_tool_calls": [],
                        "executed_tool_calls": [],
                        "parser_invalid_entries": [],
                        "invalid_actions": [],
                        "raw_model_output": {"response_text": "not json"},
                    },
                    {
                        "step_index": 2,
                        "timestamp_sec": 3.0,
                        "parser_status": VALID_OUTPUT,
                        "parser_issues": [],
                        "parser_matched_silence_rule": False,
                        "parsed_message": None,
                        "parsed_payload": {
                            "message": None,
                            "tool_calls": [{"name": "stop_reminder", "arguments": {}}],
                        },
                        "parsed_tool_calls": [
                            {"name": "stop_reminder", "arguments": {}}
                        ],
                        "requested_tool_calls": [
                            {"name": "stop_reminder", "arguments": {}}
                        ],
                        "executed_tool_calls": [],
                        "parser_invalid_entries": [],
                        "invalid_actions": [
                            {"reason": "No active overlay exists to stop."}
                        ],
                        "raw_model_output": {
                            "response_text": '{"message":null,"tool_calls":[{"name":"stop_reminder","arguments":{}}]}'
                        },
                    },
                    {
                        "step_index": 3,
                        "timestamp_sec": 5.0,
                        "parser_status": INVALID_TOOL_CALL,
                        "parser_issues": ["Unknown tool."],
                        "parser_matched_silence_rule": False,
                        "parsed_message": None,
                        "parsed_payload": {
                            "message": None,
                            "tool_calls": [{"name": "bad_tool", "arguments": {"x": 1}}],
                        },
                        "parsed_tool_calls": [],
                        "requested_tool_calls": [{"name": "bad_tool", "arguments": {"x": 1}}],
                        "executed_tool_calls": [],
                        "parser_invalid_entries": [{"name": "bad_tool"}],
                        "invalid_actions": [],
                        "raw_model_output": {"response_text": "model ``` output"},
                    },
                    {
                        "step_index": 1,
                        "timestamp_sec": 2.0,
                        "parser_status": VALID_OUTPUT,
                        "parser_issues": [],
                        "parser_matched_silence_rule": False,
                        "parsed_message": "Tracking the faucet.",
                        "parsed_payload": {
                            "message": "Tracking the faucet.",
                            "tool_calls": [
                                {"name": "start_reminder", "arguments": {"object": "faucet"}}
                            ],
                        },
                        "parsed_tool_calls": [
                            {"name": "start_reminder", "arguments": {"object": "faucet"}}
                        ],
                        "requested_tool_calls": [
                            {"name": "start_reminder", "arguments": {"object": "faucet"}}
                        ],
                        "executed_tool_calls": [
                            {"name": "start_reminder", "arguments": {"object": "faucet"}}
                        ],
                        "parser_invalid_entries": [],
                        "invalid_actions": [],
                        "raw_model_output": {
                            "response_text": '{"message":"Tracking the faucet.","tool_calls":[]}'
                        },
                    },
                ],
                "ticks": [{"tick_index": 0}, {"tick_index": 1}],
            },
        }

    def test_builds_all_four_chronological_views(self):
        memory = build_proactive_memory(self.trace_payload())

        self.assertEqual(memory["memory_schema_version"], "proact.proactive-memory.v1")
        self.assertEqual(memory["source_trace_schema_version"], "proact.clip-trace.v2")
        self.assertEqual(memory["timestamp_semantics"]["reference"], "clip-local")
        self.assertEqual(
            [event["step_index"] for event in memory["raw_model_outputs"]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(len(memory["all_output_events"]), 5)
        self.assertEqual(
            [event["step_index"] for event in memory["non_silent_output_events"]],
            [1, 2, 3, 5],
        )
        state_invalid = memory["non_silent_output_events"][1]
        self.assertEqual(state_invalid["requested_tool_calls"][0]["name"], "stop_reminder")
        self.assertEqual(state_invalid["executed_tool_calls"], [])
        self.assertEqual(
            state_invalid["invalid_actions"][0]["reason"],
            "No active overlay exists to stop.",
        )
        rejected = memory["non_silent_output_events"][2]
        self.assertEqual(rejected["original_tool_calls"][0]["arguments"], {"x": 1})
        self.assertEqual(rejected["requested_tool_calls"][0]["name"], "bad_tool")
        self.assertEqual(rejected["parser_rejected_entries"], [{"name": "bad_tool"}])
        self.assertEqual(rejected["invalid_actions"], [])
        self.assertEqual(rejected["parser_diagnostics"]["issues"], ["Unknown tool."])
        malformed = memory["non_silent_output_events"][3]
        self.assertEqual(malformed["parser_status"], UNPARSEABLE_OUTPUT)
        self.assertEqual(
            malformed["parser_diagnostics"]["issues"],
            ["Response is not valid JSON."],
        )
        self.assertNotIn("response_raw", json.dumps(memory))
        self.assertEqual(
            memory["raw_model_outputs"][2]["response_text"],
            "model ``` output",
        )

        intervals = memory["collated_tool_intervals"]
        self.assertEqual([item["instance_id"] for item in intervals], ["reminder-1", "reminder-2"])
        self.assertEqual(intervals[0]["status"], "closed")
        self.assertEqual(intervals[0]["ended_at_sec"], 5.0)
        self.assertEqual(intervals[1]["status"], "open")
        self.assertIsNone(intervals[1]["ended_at_sec"])
        self.assertEqual(intervals[1]["arguments"], {"object": "oven"})

    def test_markdown_is_rendered_from_memory_json_alone(self):
        memory = build_proactive_memory(self.trace_payload())
        round_tripped = json.loads(json.dumps(memory))
        markdown = render_proactive_memory(round_tripped)

        self.assertLess(markdown.index("reminder-1"), markdown.index("reminder-2"))
        self.assertIn("2.000s -> 5.000s", markdown)
        self.assertIn("2.000s -> OPEN", markdown)
        self.assertIn("Tracking the faucet.", markdown)
        self.assertIn('"name": "bad_tool"', markdown)
        self.assertIn("No active overlay exists to stop.", markdown)
        self.assertIn("````text\nmodel ``` output\n````", markdown)
        raw_section = markdown.index("## Raw Model Outputs")
        self.assertLess(
            markdown.index("step 1", raw_section),
            markdown.index("step 3", raw_section),
        )

    def test_ticks_without_outputs_do_not_create_memory_events(self):
        memory = build_proactive_memory(
            {
                "trace_schema_version": "proact.clip-trace.v2",
                "clip": {"clip_uid": "silent"},
                "trace": {
                    "trace_schema_version": "proact.clip-trace.v2",
                    "predicted_intervals": [],
                    "output_events": [],
                    "ticks": [{"tick_index": 0}],
                },
            }
        )
        self.assertEqual(memory["raw_model_outputs"], [])
        self.assertEqual(memory["all_output_events"], [])
        markdown = render_proactive_memory(memory)
        self.assertIn("_No model-output events._", markdown)
        self.assertNotIn("### 0.000s", markdown)


class ClipRunnerTests(unittest.TestCase):
    def test_writes_detailed_traces_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base)
            config = make_config(base, prompt, annotation)
            model = StubModelRunner([
                '{"message": "tracking", "tool_calls": [{"name": "start_reminder", "arguments": {"object": "faucet"}}]}',
                '{"message": null, "tool_calls": []}',
            ])
            runner = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=model,
                video_sampler=StubSampler(),
            )
            result = runner.run()
            self.assertEqual(result["completed_clip_count"], 2)
            self.assertEqual(result["failed_clip_count"], 0)

            shard_root = base / "run" / "shards" / "shard-00000-of-00001"
            clip_dir = shard_root / "clips" / "clip-0"
            trace_path = clip_dir / "trace.json"
            memory_json_path = clip_dir / "proactive_memory.json"
            memory_markdown_path = clip_dir / "proactive_memory.md"
            manifest_path = clip_dir / "manifest.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            memory = json.loads(memory_json_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trace = payload["trace"]
            self.assertEqual(payload["clip"]["input_video_path"], str(clip_root / "clip-0.mp4"))
            self.assertEqual(len(trace["ticks"]), 1)
            self.assertEqual(len(trace["output_events"]), 1)
            event = trace["output_events"][0]
            self.assertEqual(event["parsed_message"], "tracking")
            self.assertEqual(event["requested_tool_calls"][0]["name"], "start_reminder")
            self.assertEqual(event["executed_tool_calls"][0]["name"], "start_reminder")
            self.assertTrue(trace["predicted_intervals"][0]["is_open"])
            self.assertEqual(payload["trace_schema_version"], "proact.clip-trace.v2")
            self.assertEqual(trace["trace_schema_version"], "proact.clip-trace.v2")
            self.assertFalse((shard_root / "traces").exists())
            self.assertTrue(memory_json_path.is_file())
            self.assertTrue(memory_markdown_path.is_file())
            self.assertEqual(memory["memory_schema_version"], "proact.proactive-memory.v1")
            self.assertIn(
                "## Collated Memory",
                memory_markdown_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                manifest["artifacts"]["trace"]["sha256"],
                hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["artifacts"]["proactive_memory_json"]["sha256"],
                hashlib.sha256(memory_json_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["artifacts"]["proactive_memory_markdown"]["sha256"],
                hashlib.sha256(memory_markdown_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["clip_artifacts"][0]["path"], "clips/clip-0")
            self.assertEqual(
                result["clip_artifacts"][0]["proactive_memory_json"],
                "clips/clip-0/proactive_memory.json",
            )
            self.assertEqual(
                result["clip_artifacts"][0]["proactive_memory_markdown"],
                "clips/clip-0/proactive_memory.md",
            )
            self.assertNotIn("samples", payload)
            self.assertNotIn("metrics", payload)
            self.assertNotIn("image_b64", json.dumps(payload))

            resumed = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=True,
                model_runner=StubModelRunner([]),
                video_sampler=StubSampler(),
            ).run()
            self.assertEqual(resumed["resumed_clip_count"], 2)
            self.assertEqual(resumed["repaired_artifact_count"], 0)
            self.assertEqual(resumed["completed_clip_count"], 0)

    def test_resume_repairs_memory_without_model_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "remember this", "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            ).run()

            clip_dir = (
                base / "run" / "shards" / "shard-00000-of-00001" / "clips" / "clip-0"
            )
            memory_json_path = clip_dir / "proactive_memory.json"
            memory_path = clip_dir / "proactive_memory.md"
            manifest_path = clip_dir / "manifest.json"
            memory = json.loads(memory_json_path.read_text(encoding="utf-8"))
            memory["all_output_events"][0]["message"] = "rendered only from memory JSON"
            memory_json_path.write_text(
                json.dumps(memory, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["proactive_memory_json"]["sha256"] = hashlib.sha256(
                memory_json_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            memory_path.write_text("corrupt", encoding="utf-8")
            model = StubModelRunner([])
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=True,
                model_runner=model,
                video_sampler=StubSampler(),
            ).run()

            self.assertEqual(result["repaired_artifact_count"], 1)
            self.assertEqual(result["resumed_clip_count"], 0)
            self.assertEqual(result["completed_clip_count"], 0)
            self.assertEqual(model.calls, [])
            markdown = memory_path.read_text(encoding="utf-8")
            self.assertIn("rendered only from memory JSON", markdown)
            self.assertNotIn(
                "remember this",
                markdown[:markdown.index("## Raw Model Outputs")],
            )
            manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifacts"]["proactive_memory_markdown"]["sha256"],
                hashlib.sha256(memory_path.read_bytes()).hexdigest(),
            )

    def test_resume_rebuilds_corrupt_memory_json_and_markdown_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "source trace message", "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            ).run()

            clip_dir = (
                base / "run" / "shards" / "shard-00000-of-00001" / "clips" / "clip-0"
            )
            memory_json_path = clip_dir / "proactive_memory.json"
            memory_markdown_path = clip_dir / "proactive_memory.md"
            memory_json_path.write_text('{"corrupt": true}\n', encoding="utf-8")
            memory_markdown_path.write_text("stale", encoding="utf-8")
            model = StubModelRunner([])
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=True,
                model_runner=model,
                video_sampler=StubSampler(),
            ).run()

            self.assertEqual(result["repaired_artifact_count"], 1)
            self.assertEqual(model.calls, [])
            memory = json.loads(memory_json_path.read_text(encoding="utf-8"))
            self.assertEqual(memory["memory_schema_version"], "proact.proactive-memory.v1")
            self.assertEqual(memory["all_output_events"][0]["message"], "source trace message")
            self.assertIn(
                "source trace message",
                memory_markdown_path.read_text(encoding="utf-8"),
            )

    def test_resume_repairs_existing_v2_bundle_without_memory_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "legacy v2", "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            ).run()

            clip_dir = (
                base / "run" / "shards" / "shard-00000-of-00001" / "clips" / "clip-0"
            )
            memory_json_path = clip_dir / "proactive_memory.json"
            memory_json_path.unlink()
            manifest_path = clip_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            markdown_record = manifest["artifacts"].pop("proactive_memory_markdown")
            manifest["artifacts"].pop("proactive_memory_json")
            manifest["artifacts"]["proactive_memory"] = markdown_record
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            model = StubModelRunner([])
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=True,
                model_runner=model,
                video_sampler=StubSampler(),
            ).run()

            self.assertEqual(result["repaired_artifact_count"], 1)
            self.assertEqual(model.calls, [])
            memory = json.loads(memory_json_path.read_text(encoding="utf-8"))
            self.assertEqual(memory["all_output_events"][0]["message"], "legacy v2")
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("proactive_memory_json", repaired_manifest["artifacts"])
            self.assertIn("proactive_memory_markdown", repaired_manifest["artifacts"])
            self.assertNotIn("proactive_memory", repaired_manifest["artifacts"])

    def test_resume_reruns_when_trace_hash_is_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "first", "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            ).run()

            trace_path = (
                base / "run" / "shards" / "shard-00000-of-00001"
                / "clips" / "clip-0" / "trace.json"
            )
            trace_path.write_text(
                trace_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            model = StubModelRunner([
                '{"message": "rerun", "tool_calls": []}',
            ])
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=True,
                model_runner=model,
                video_sampler=StubSampler(),
            ).run()

            self.assertEqual(result["completed_clip_count"], 1)
            self.assertEqual(result["repaired_artifact_count"], 0)
            self.assertEqual(len(model.calls), 1)
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["trace"]["output_events"][0]["parsed_message"], "rerun")

    def test_failed_forced_rerun_removes_stale_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=StubModelRunner([
                    '{"message": "complete", "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            ).run()

            clip_dir = (
                base / "run" / "shards" / "shard-00000-of-00001" / "clips" / "clip-0"
            )
            result = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                resume=False,
                model_runner=StubModelRunner([]),
                video_sampler=StubSampler(),
            ).run()

            self.assertEqual(result["failed_clip_count"], 1)
            self.assertFalse((clip_dir / "manifest.json").exists())
            self.assertFalse((clip_dir / "trace.json").exists())
            self.assertFalse((clip_dir / "proactive_memory.json").exists())
            self.assertFalse((clip_dir / "proactive_memory.md").exists())

    def test_failure_is_isolated_and_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base)
            (clip_root / "clip-1.mp4").unlink()
            config = make_config(base, prompt, annotation)
            runner = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                max_clip_attempts=2,
                model_runner=StubModelRunner([
                    '{"message": null, "tool_calls": []}',
                ]),
                video_sampler=StubSampler(),
            )
            result = runner.run()
            self.assertEqual(result["completed_clip_count"], 1)
            self.assertEqual(result["failed_clip_count"], 1)
            failure = json.loads((
                base / "run" / "shards" / "shard-00000-of-00001" / "failures" / "clip-1.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(failure["attempts"], 2)
            self.assertEqual(failure["error_type"], "FileNotFoundError")

    def test_provider_error_fails_clip_into_failures_never_scored(self):
        # A provider/API error (rate limit, 5xx, adapter crash) must NEVER be
        # parsed as model output and scored as silence: the clip fails loudly
        # and lands in failures/<uid>.json with an exception report.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)

            class ErrorModelRunner:
                prompt_builder = None

                def infer(self, **kwargs):
                    return {
                        "request_messages": [],
                        "response_ok": False,
                        "response_text": "",
                        "response_reasoning": None,
                        "response_finish_reason": None,
                        "response_error": "429 rate limited",
                        "response_error_type": "HTTPError",
                        "response_status_code": 429,
                        "response_raw": None,
                    }

            runner = ClipTraceRunner(
                config,
                clip_root=str(clip_root),
                model_runner=ErrorModelRunner(),
                video_sampler=StubSampler(),
            )
            result = runner.run()

            self.assertEqual(result["completed_clip_count"], 0)
            self.assertEqual(result["failed_clip_count"], 1)
            shard_root = base / "run" / "shards" / "shard-00000-of-00001"
            failure = json.loads(
                (shard_root / "failures" / "clip-0.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["error_type"], "ModelInferenceError")
            self.assertIn("429", failure["error"])
            self.assertFalse((shard_root / "clips" / "clip-0" / "trace.json").exists())

    def test_run_fingerprint_key_allows_compatible_path_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root_a = write_fixture(base, clip_count=1)
            clip_root_b = base / "clips-b"
            clip_root_b.mkdir()
            (clip_root_b / "clip-0.mp4").write_bytes(
                (clip_root_a / "clip-0.mp4").read_bytes()
            )
            config = make_config(base, prompt, annotation)
            key = "video-capture-llava-onevision-run-2026-06-12"

            first = ClipTraceRunner(
                config,
                clip_root=str(clip_root_a),
                run_fingerprint_key=key,
                model_runner=StubModelRunner([
                    "{\"message\": null, \"tool_calls\": []}",
                ]),
                video_sampler=StubSampler(),
            )
            first_effective = first._run_fingerprint()
            first_natural = first._natural_run_fingerprint()
            first.run()

            second_model = StubModelRunner([])
            second = ClipTraceRunner(
                config,
                clip_root=str(clip_root_b),
                run_fingerprint_key=key,
                resume=True,
                model_runner=second_model,
                video_sampler=StubSampler(),
            )
            self.assertEqual(second._run_fingerprint(), first_effective)
            # The natural fingerprint is PORTABLE: relocating the same clips
            # (e.g. a different $TMPDIR) must not change run identity.
            self.assertEqual(second._natural_run_fingerprint(), first_natural)
            result = second.run()
            self.assertEqual(result["resumed_clip_count"], 1)
            self.assertEqual(second_model.calls, [])

            manifest_text = (base / "run" / "run_manifest.json").read_text(
                encoding="utf-8"
            )
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["run_fingerprint_mode"], "key")
            self.assertEqual(
                manifest["run_fingerprint_key_sha256"],
                hashlib.sha256(key.encode("utf-8")).hexdigest(),
            )
            self.assertNotIn(key, manifest_text)

            with self.assertRaisesRegex(ValueError, "incompatible run_fingerprint"):
                ClipTraceRunner(
                    config,
                    clip_root=str(clip_root_b),
                    run_fingerprint_key="different-run-key",
                    model_runner=StubModelRunner([]),
                    video_sampler=StubSampler(),
                ).run()

    def test_run_fingerprint_key_rejects_blank_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prompt, annotation, clip_root = write_fixture(base, clip_count=1)
            config = make_config(base, prompt, annotation)
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                ClipTraceRunner(
                    config,
                    clip_root=str(clip_root),
                    run_fingerprint_key="   ",
                    model_runner=StubModelRunner([]),
                    video_sampler=StubSampler(),
                )

    def test_sharding_is_stable_and_disjoint(self):
        assignments = [[uid for uid in ("a", "b", "c", "d") if _shard_for_clip(uid, 3) == shard] for shard in range(3)]
        flattened = [uid for shard in assignments for uid in shard]
        self.assertEqual(sorted(flattened), ["a", "b", "c", "d"])
        self.assertEqual(_shard_for_clip("a", 3), _shard_for_clip("a", 3))


class AdapterAndMediaTests(unittest.TestCase):
    def test_every_configured_adapter_factory_builds(self):
        for name in (
            "minicpm-o45",
            "llava-onevision15",
            "videollama3",
        ):
            self.assertIsNotNone(build_vidllm_adapter(name, {}))
        for name in ("videollm-online", "mmduet"):
            self.assertIsNotNone(build_streaming_adapter(name))

    def test_frame_arrays_are_owned_across_clip_and_session_boundaries(self):
        sampler = DefaultVideoSampler(SimpleNamespace(resize_frames=False))
        reused = np.zeros((2, 2, 3), dtype=np.uint8)
        reused[:, :, 0] = 255
        first = DummyFrame(0.0, reused)
        sampler._prepare_frames([first])
        reused[:, :, :] = 0
        reused[:, :, 1] = 255
        self.assertTrue(np.all(first.array[:, :, 0] == 255))
        self.assertTrue(np.all(first.array[:, :, 1] == 0))

        class ConcreteBufferedSession(BufferedStreamingSession):
            def finalize(self):
                return []

        session = ConcreteBufferedSession(
            input_video_path="clip.mp4",
            frame_fps=1.0,
            adapter_settings={},
            trace_metadata={},
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        session.ingest_frame(frame=frame, timestamp_sec=0.0)
        frame[:, :, :] = 99
        self.assertTrue(np.all(session.frames[0] == 0))
        self.assertTrue(session.frames[0].flags.owndata)


if __name__ == "__main__":
    unittest.main()
