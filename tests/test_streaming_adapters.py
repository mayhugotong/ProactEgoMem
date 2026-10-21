from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.support import install_common_stubs


install_common_stubs()

from proact.evaluation.runtime.streaming.adapters import (
    MMDuetSession,
    VideoLLMOnlineSession,
)


class StreamingAdapterTests(unittest.TestCase):
    def test_videollm_online_session_coerces_batchencoding_like_token_outputs(self) -> None:
        class FakeBatchEncoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def to(self, device):
                return self

        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return FakeBatchEncoding([[1, 2, 3]])

        infer = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            _start_ids=FakeBatchEncoding([[4, 5]]),
            _added_stream_prompt_ids=FakeBatchEncoding([[6]]),
            _added_stream_generation_ids=FakeBatchEncoding([[7]]),
            last_ids=FakeBatchEncoding([[8]]),
        )

        session = VideoLLMOnlineSession(
            input_video_path="/tmp/video.mp4",
            frame_fps=1.0,
            adapter_settings={},
            trace_metadata={},
        )

        session._patch_liveinfer_token_state(infer)

        self.assertEqual(infer._start_ids, [[4, 5]])
        self.assertEqual(infer._added_stream_prompt_ids, [[6]])
        self.assertEqual(infer._added_stream_generation_ids, [[7]])
        self.assertEqual(infer.last_ids, [[8]])
        self.assertEqual(infer.tokenizer.apply_chat_template([]), [[1, 2, 3]])

    def test_videollm_online_session_extracts_timestamped_events(self) -> None:
        class FakeLiveInfer:
            def __init__(self):
                self.model = SimpleNamespace(config=SimpleNamespace(frame_resolution=1))
                self.query_calls = []

            def reset(self):
                return None

            def load_video(self, path):
                raise AssertionError("VideoLLMOnlineSession should use sampled frames, not reload the raw video.")

            def input_query_stream(self, text, video_time=None):
                self.query_calls.append((text, video_time))

            def input_video_stream(self, timestamp_sec):
                self.current_time = timestamp_sec

            def __call__(self):
                return None, (
                    f'(Video Time = {self.current_time}s) Assistant: '
                    '{"tool": {"name": "start_overlay", "args": {"name": "lamp", "state": "on"}}}'
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = VideoLLMOnlineSession(
                input_video_path=str(Path(tmp_dir) / "video.mp4"),
                frame_fps=2.0,
                adapter_settings={"repo_root": tmp_dir},
                trace_metadata={},
            )
            session.inject_text(text="Prompt", timestamp_sec=0.0)
            session.inject_text(text="State", timestamp_sec=4.0)
            session.ingest_frame(frame=[[[0, 0, 0]]], timestamp_sec=1.5)
            session.ingest_frame(frame=[[[0, 0, 0]]], timestamp_sec=4.0)

            fake_module = SimpleNamespace(LiveInfer=FakeLiveInfer)
            with mock.patch("importlib.import_module", return_value=fake_module):
                with mock.patch.object(
                    VideoLLMOnlineSession,
                    "_sampled_frames_to_model_video_tensor",
                    return_value=[[[[0]]]],
                ):
                    events = session.finalize()
                    repeated_events = session.finalize()

        self.assertEqual(len(events), 2)
        self.assertEqual(repeated_events, [])
        self.assertEqual(events[0].timestamp_sec, 1.5)
        self.assertEqual(events[1].timestamp_sec, 4.0)
        self.assertIn("start_overlay", events[0].response_text)
        self.assertEqual(events[0].raw_model_output["adapter_debug"]["synthetic_frame_time"], 0.0)
        self.assertEqual(events[1].raw_model_output["adapter_debug"]["synthetic_frame_time"], 0.5)

    def test_mmduet_session_normalizes_benchmark_responses(self) -> None:
        class FakeArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeInfer:
            def __init__(self, args):
                self.args = args
                self.debug_data_list = [{"score": 0.9}]
                self.queries = []
                self.frame_count = 0

            def reset(self):
                return None

            def set_fps(self, fps):
                self.fps = fps

            def input_video_stream(self, frames):
                self.frames = frames

            def encode_given_query(self, text):
                self.queries.append(text)

            def input_one_frame(self):
                self.frame_count += 1
                if self.frame_count == 1:
                    return {
                        "time": 5.0,
                        "response": '{"tool": {"name": "start_overlay", "args": {"name": "door", "state": "open"}}}',
                    }
                if self.frame_count == 2:
                    return {
                        "time": 6.0,
                        "response": '{"tool": {"name": "stop_overlay", "args": {"id": "overlay-1"}}}',
                    }
                return {"time": 7.0, "response": None}

        def fake_import(name):
            if name == "models.arguments_live":
                return SimpleNamespace(get_args_class=lambda live_version: FakeArgs)
            if name == "demo.liveinfer":
                return SimpleNamespace(LiveInferForDemo=FakeInfer)
            raise AssertionError(name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = MMDuetSession(
                input_video_path=str(Path(tmp_dir) / "video.mp4"),
                frame_fps=1.0,
                adapter_settings={"repo_root": tmp_dir, "llm_pretrained": "llava-live"},
                trace_metadata={},
            )
            session.inject_text(text="Prompt", timestamp_sec=0.0)
            events = []

            with mock.patch("importlib.import_module", side_effect=fake_import):
                events.extend(session.ingest_frame(frame=[[[0]]], timestamp_sec=1.0))
                session.inject_text(text="State update", timestamp_sec=5.0)
                events.extend(session.ingest_frame(frame=[[[1]]], timestamp_sec=6.0))
                events.extend(session.finalize())

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].timestamp_sec, 5.0)
        self.assertEqual(events[0].raw_model_output["debug_data"], [{"score": 0.9}])
        self.assertEqual([m["content"] for m in events[0].request_messages], ["Prompt"])
        self.assertEqual([m["content"] for m in events[1].request_messages], ["Prompt", "State update"])

