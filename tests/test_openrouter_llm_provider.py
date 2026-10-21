from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.support import install_common_stubs

install_common_stubs()

from proact.models.llm.base import ImageB64Part, LLMMessage, LLMRequest, TextPart
from proact.models.llm.registry import create_llm


class OpenRouterLLMProviderTests(unittest.TestCase):
    def test_builds_multimodal_json_chat_completion_request(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(*, url, headers, payload, timeout_s):
            captured.update(url=url, headers=headers, payload=payload, timeout_s=timeout_s)
            return {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"message":null,"tool_calls":[]}',
                    },
                }],
                "usage": {"prompt_tokens": 17, "completion_tokens": 8},
            }

        with patch(
            "proact.models.llm.providers.openrouter._post_json",
            side_effect=fake_post_json,
        ):
            llm = create_llm(
                provider="openrouter",
                model="google/gemini-3.5-flash",
                api_base="https://openrouter.ai/api/v1",
                api_key="openrouter-secret",
                request_timeout_s=90,
                params={"temperature": 0.2, "max_tokens": 128, "image_detail": "low"},
                extra={
                    "http_referer": "https://example.org/proact",
                    "app_title": "ProAct Benchmark",
                },
            )
            response = llm.complete_one(LLMRequest(
                messages=[
                    LLMMessage(role="system", content="Return JSON only."),
                    LLMMessage(role="user", content=[
                        TextPart(text="Inspect this frame."),
                        ImageB64Part(
                            b64="ZmFrZQ==",
                            mime_type="image/png",
                            alt="frame #1",
                        ),
                    ]),
                ],
                json_mode=True,
            ))

        self.assertTrue(response.ok)
        self.assertEqual(response.text, '{"message":null,"tool_calls":[]}')
        self.assertEqual(response.usage_tokens_prompt, 17)
        self.assertEqual(response.usage_tokens_completion, 8)
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        headers = captured["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["Authorization"], "Bearer openrouter-secret")
        self.assertEqual(headers["HTTP-Referer"], "https://example.org/proact")
        self.assertEqual(headers["X-OpenRouter-Title"], "ProAct Benchmark")
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["model"], "google/gemini-3.5-flash")
        self.assertEqual(payload["messages"][0]["role"], "system")
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "Inspect this frame."})
        self.assertEqual(user_content[1]["image_url"]["detail"], "low")
        self.assertEqual(
            user_content[1]["image_url"]["url"],
            "data:image/png;base64,ZmFrZQ==",
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 128)
        self.assertNotIn("image_detail", payload)

    def test_requires_api_key_without_sending_request(self) -> None:
        llm = create_llm(
            provider="openrouter",
            model="openai/gpt-5.4",
            api_base="https://openrouter.ai/api/v1",
        )
        with patch("proact.models.llm.providers.openrouter._post_json") as post:
            response = llm.complete_one(LLMRequest(
                messages=[LLMMessage(role="user", content="Return JSON only.")]
            ))

        self.assertFalse(response.ok)
        self.assertEqual(response.error_type, "ValueError")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
