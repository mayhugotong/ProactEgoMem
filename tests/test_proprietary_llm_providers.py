from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.support import install_common_stubs


install_common_stubs()

from proact.models.llm.base import ImageB64Part, LLMMessage, LLMRequest, TextPart
from proact.models.llm.registry import create_llm


class ProprietaryLLMProviderTests(unittest.TestCase):
    def test_gemini_provider_builds_multimodal_generate_content_request(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(*, url, headers, payload, timeout_s):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            captured["timeout_s"] = timeout_s
            return {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [{"text": '{"skip":true}'}],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 5,
                },
            }

        with patch("proact.models.llm.providers.gemini._post_json", side_effect=fake_post_json):
            llm = create_llm(
                provider="gemini",
                model="gemini-2.5-pro",
                api_base="https://generativelanguage.googleapis.com/v1beta",
                api_key="gemini-secret",
                params={"temperature": 0.1, "max_output_tokens": 64},
            )
            response = llm.complete_one(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content="System guidance"),
                        LLMMessage(
                            role="user",
                            content=[
                                TextPart(text="Look at the frame."),
                                ImageB64Part(b64="ZmFrZQ==", mime_type="image/png", alt="frame #1"),
                            ],
                        ),
                    ],
                    json_mode=True,
                )
            )

        self.assertTrue(response.ok)
        self.assertEqual(response.text, '{"skip":true}')
        self.assertEqual(captured["url"], "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent")
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["system_instruction"]["parts"][0]["text"], "System guidance")
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Look at the frame.")
        self.assertEqual(payload["contents"][0]["parts"][1]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 64)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_anthropic_provider_builds_multimodal_messages_request(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(*, url, headers, payload, timeout_s):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            captured["timeout_s"] = timeout_s
            return {
                "content": [{"type": "text", "text": '{"tool":{"name":"stop_reminder","args":{"id":"reminder-1"}}}'}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 21,
                    "output_tokens": 9,
                },
            }

        with (
            patch("proact.models.llm.providers.anthropic._post_json", side_effect=fake_post_json),
            patch(
                "proact.models.llm.providers.anthropic._transcode_image_b64_part",
                return_value=("jpeg-b64", "image/jpeg"),
            ),
        ):
            llm = create_llm(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_base="https://api.anthropic.com",
                api_key="anthropic-secret",
                params={
                    "temperature": 0.2,
                    "max_tokens": 128,
                    "image_format": "jpeg",
                    "image_quality": 80,
                },
                extra={"anthropic_version": "2023-06-01"},
            )
            response = llm.complete_one(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content="Be concise."),
                        LLMMessage(
                            role="user",
                            content=[
                                TextPart(text="Decide on the overlay action."),
                                ImageB64Part(b64="png-b64", mime_type="image/png", alt="frame #1"),
                            ],
                        ),
                    ]
                )
            )

        self.assertTrue(response.ok)
        self.assertIn("stop_reminder", response.text or "")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["system"], "Be concise.")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "Decide on the overlay action.")
        self.assertEqual(payload["messages"][0]["content"][1]["source"]["media_type"], "image/jpeg")
        self.assertEqual(payload["max_tokens"], 128)

    def test_openai_provider_forwards_reasoning_and_text_params(self) -> None:
        captured: dict[str, object] = {}

        class FakeUsage:
            input_tokens = 12
            output_tokens = 7

        class FakeResponse:
            output_text = '{"skip":true}'
            usage = FakeUsage()
            status = "completed"

            def model_dump(self):
                return {"id": "resp-1"}

        class FakeResponsesAPI:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            responses = FakeResponsesAPI()

        with (
            patch("proact.models.llm.providers.openai.OpenAIResponses._get_client", return_value=FakeClient()),
            patch(
                "proact.models.llm.providers.openai._transcode_image_b64_part",
                return_value=("jpeg-b64", "image/jpeg"),
            ),
        ):
            llm = create_llm(
                provider="openai",
                model="gpt-5.2",
                api_base="https://api.openai.com/v1",
                api_key="openai-secret",
                params={
                    "reasoning": {"effort": "low"},
                    "text": {"verbosity": "low"},
                    "max_output_tokens": 512,
                },
            )
            response = llm.complete_one(
                LLMRequest(
                    messages=[LLMMessage(role="user", content="Return JSON only.")]
                )
            )

        self.assertTrue(response.ok)
        self.assertEqual(captured["model"], "gpt-5.2")
        self.assertEqual(captured["reasoning"], {"effort": "low"})
        self.assertEqual(captured["text"], {"verbosity": "low", "format": {"type": "text"}})
        self.assertEqual(captured["max_output_tokens"], 512)
        self.assertEqual(captured["timeout"], 60.0)

    def test_openai_provider_uses_json_mode_and_configured_image_detail(self) -> None:
        captured: dict[str, object] = {}

        class FakeUsage:
            input_tokens = 12
            output_tokens = 7

        class FakeResponse:
            output_text = '{"act":false,"tool_name":null,"arguments":{}}'
            usage = FakeUsage()
            status = "completed"

            def model_dump(self):
                return {"id": "resp-1"}

        class FakeResponsesAPI:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            responses = FakeResponsesAPI()

        with (
            patch("proact.models.llm.providers.openai.OpenAIResponses._get_client", return_value=FakeClient()),
            patch(
                "proact.models.llm.providers.openai._transcode_image_b64_part",
                return_value=("jpeg-b64", "image/jpeg"),
            ),
        ):
            llm = create_llm(
                provider="openai",
                model="gpt-5.4-nano",
                api_base="https://api.openai.com/v1",
                api_key="openai-secret",
                params={
                    "reasoning": {"effort": "none"},
                    "text": {"verbosity": "low"},
                    "image_detail": "low",
                    "image_format": "jpeg",
                    "image_quality": 80,
                    "max_output_tokens": 128,
                },
            )
            response = llm.complete_one(
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role="user",
                            content=[
                                TextPart(text="Return JSON only."),
                                ImageB64Part(b64="png-b64", mime_type="image/png", alt="frame #1"),
                            ],
                        )
                    ],
                    json_mode=True,
                )
            )

        self.assertTrue(response.ok)
        self.assertEqual(captured["model"], "gpt-5.4-nano")
        self.assertEqual(captured["text"], {"verbosity": "low", "format": {"type": "json_object"}})
        self.assertEqual(captured["max_output_tokens"], 128)
        input_items = captured["input"]
        assert isinstance(input_items, list)
        content = input_items[0]["content"]
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["detail"], "low")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))
