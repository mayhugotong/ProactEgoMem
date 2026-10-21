from __future__ import annotations

import unittest

from tests.support import install_common_stubs


install_common_stubs()

from proact.evaluation.runtime.model_runner import LLMModelRunner
from proact.models.llm.base import LLMMessage, LLMResponse, TextPart


class FakePromptBuilder:
    def build_messages(self, **_kwargs):
        return [LLMMessage(role="user", content=[TextPart(text="Return JSON only.")])]


class FakeLLM:
    def __init__(self):
        self.requests = []

    def complete_one(self, request):
        self.requests.append(request)
        return LLMResponse(text='{"act": false, "tool_name": null, "arguments": {}}')


class LLMModelRunnerTests(unittest.TestCase):
    def test_request_params_can_enable_json_mode_without_forwarding_as_param(self) -> None:
        llm = FakeLLM()
        runner = LLMModelRunner(
            llm=llm,
            prompt_builder=FakePromptBuilder(),
            request_params={"json_mode": True, "max_output_tokens": 128},
        )

        runner.infer(
            frames=[],
            tool_state={},
            step_index=0,
            timestamp_sec=0.0,
            evaluation_context="short-video",
            input_mode="sliding-window",
            input_video_path="/tmp/video.mp4",
            metadata={},
        )

        self.assertEqual(len(llm.requests), 1)
        request = llm.requests[0]
        self.assertTrue(request.json_mode)
        self.assertEqual(request.params, {"max_output_tokens": 128})


if __name__ == "__main__":
    unittest.main()
