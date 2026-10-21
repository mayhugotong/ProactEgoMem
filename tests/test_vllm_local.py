from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.support import install_common_stubs


install_common_stubs()

from proact.models.llm.base import LLMMessage, LLMRequest
from proact.models.llm.providers import vllm_local


class VLLMLocalTests(unittest.TestCase):
    def make_provider(self, *, params=None, extra=None):
        provider = object.__new__(vllm_local.VLLMLocal)
        provider.config = SimpleNamespace(
            model="demo-model",
            params=params or {},
            extra=extra or {},
        )
        provider._lock = threading.RLock()
        provider._busy = 0
        provider._engine = SimpleNamespace(model="demo-model")
        return provider

    def test_prepare_request_passes_only_provided_sampling_params(self) -> None:
        provider = self.make_provider(
            params={"top_p": 0.9, "temperature": None},
            extra={"chat_template_kwargs": {"enable_thinking": True}},
        )
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="hello")],
            params={
                "max_tokens": 128,
                "presence_penalty": 0.3,
                "model": "demo-model",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(
            prepared["sampling_kwargs"],
            {"top_p": 0.9, "max_tokens": 128, "presence_penalty": 0.3},
        )
        self.assertEqual(prepared["chat_template_kwargs"], {"enable_thinking": False})

    def test_group_requests_uses_effective_sampling_and_template_config(self) -> None:
        provider = self.make_provider()
        prepared_requests = [
            {
                "messages": [{"role": "user", "content": "a"}],
                "sampling_kwargs": {"temperature": 0.2},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {
                "messages": [{"role": "user", "content": "b"}],
                "sampling_kwargs": {"temperature": 0.2},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {
                "messages": [{"role": "user", "content": "c"}],
                "sampling_kwargs": {"temperature": 0.7},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ]

        grouped = provider._group_requests(prepared_requests)
        group_sizes = sorted(len(group["indices"]) for group in grouped.values())

        self.assertEqual(group_sizes, [1, 2])

    def test_complete_surfaces_sampling_param_errors(self) -> None:
        provider = self.make_provider(params={"unsupported_flag": "boom"})
        request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])

        with mock.patch.object(
            vllm_local,
            "SamplingParams",
            side_effect=TypeError("unsupported sampling parameter"),
        ):
            responses = provider.complete([request])

        self.assertEqual(len(responses), 1)
        self.assertFalse(responses[0].ok)
        self.assertEqual(responses[0].error_type, "TypeError")
        self.assertIn("unsupported sampling parameter", responses[0].error)

    def test_request_chat_template_kwargs_fall_back_to_config(self) -> None:
        provider = self.make_provider(
            params={"temperature": 0.2},
            extra={"chat_template_kwargs": {"enable_thinking": True}},
        )
        request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template_kwargs"], {"enable_thinking": True})

    def test_request_chat_template_content_format_falls_back_to_config(self) -> None:
        provider = self.make_provider(extra={"chat_template_content_format": "openai"})
        request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template_content_format"], "openai")
        self.assertNotIn("chat_template_content_format", prepared["sampling_kwargs"])

    def test_request_chat_template_content_format_param_overrides_config(self) -> None:
        provider = self.make_provider(extra={"chat_template_content_format": "string"})
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="hello")],
            params={"chat_template_content_format": "openai"},
        )

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template_content_format"], "openai")

    def test_request_chat_template_can_be_configured(self) -> None:
        provider = self.make_provider(
            extra={
                "chat_template": "configured-template",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
        request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template"], "configured-template")
        self.assertEqual(prepared["chat_template_kwargs"], {"enable_thinking": True})

    def test_request_chat_template_path_can_be_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "chat_template.jinja"
            template_path.write_text("configured-template-from-file", encoding="utf-8")
            provider = self.make_provider(extra={"chat_template_path": str(template_path)})
            request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])

            prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template"], "configured-template-from-file")

    def test_request_chat_template_param_overrides_config(self) -> None:
        provider = self.make_provider(extra={"chat_template": "configured-template"})
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="hello")],
            params={"chat_template": "request-template"},
        )

        prepared = provider._prepare_request(request, loaded_model="demo-model")

        self.assertEqual(prepared["chat_template"], "request-template")

    def test_execute_group_fails_when_chat_template_is_missing(self) -> None:
        provider = self.make_provider()
        tokenizer = SimpleNamespace(chat_template=None)
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            generation = SimpleNamespace(text="{}", finish_reason="stop", token_ids=[1])
            return [SimpleNamespace(outputs=[generation], prompt_token_ids=[1, 2])]

        engine = SimpleNamespace(
            has_chat=True,
            tokenizer=tokenizer,
            vllm=SimpleNamespace(chat=fake_chat),
        )

        with self.assertRaisesRegex(ValueError, "tokenizer does not define chat_template"):
            responses = provider._execute_group(
                engine=engine,
                messages_list=[[{"role": "user", "content": "hello"}]],
                sampling_params=SimpleNamespace(),
                chat_template=None,
                chat_template_kwargs={},
                chat_template_content_format=None,
            )
        self.assertEqual(captured, {})

    def test_execute_group_keeps_tokenizer_chat_template_when_present(self) -> None:
        provider = self.make_provider()
        tokenizer = SimpleNamespace(chat_template="tokenizer-template")
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            generation = SimpleNamespace(text="{}", finish_reason="stop", token_ids=[1])
            return [SimpleNamespace(outputs=[generation], prompt_token_ids=[1, 2])]

        engine = SimpleNamespace(
            has_chat=True,
            tokenizer=tokenizer,
            vllm=SimpleNamespace(chat=fake_chat),
        )

        responses = provider._execute_group(
            engine=engine,
            messages_list=[[{"role": "user", "content": "hello"}]],
            sampling_params=SimpleNamespace(),
            chat_template=None,
            chat_template_kwargs={},
            chat_template_content_format=None,
        )

        self.assertTrue(responses[0].ok)
        self.assertNotIn("chat_template", captured)

    def test_execute_group_configured_template_overrides_tokenizer_default(self) -> None:
        provider = self.make_provider()
        tokenizer = SimpleNamespace(chat_template="tokenizer-template")
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            generation = SimpleNamespace(text="{}", finish_reason="stop", token_ids=[1])
            return [SimpleNamespace(outputs=[generation], prompt_token_ids=[1, 2])]

        engine = SimpleNamespace(
            has_chat=True,
            tokenizer=tokenizer,
            vllm=SimpleNamespace(chat=fake_chat),
        )

        responses = provider._execute_group(
            engine=engine,
            messages_list=[[{"role": "user", "content": "hello"}]],
            sampling_params=SimpleNamespace(),
            chat_template="fallback-template",
            chat_template_kwargs={},
            chat_template_content_format="openai",
        )

        self.assertTrue(responses[0].ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["chat_template"], "fallback-template")
        self.assertEqual(calls[0]["chat_template_content_format"], "openai")

    def test_execute_group_explicit_content_format_type_error_is_not_silenced(self) -> None:
        provider = self.make_provider()
        tokenizer = SimpleNamespace(chat_template="tokenizer-template")

        def fake_chat(**_kwargs):
            raise TypeError("bad chat_template_content_format")

        engine = SimpleNamespace(
            has_chat=True,
            tokenizer=tokenizer,
            vllm=SimpleNamespace(chat=fake_chat),
        )

        with self.assertRaisesRegex(TypeError, "bad chat_template_content_format"):
            provider._execute_group(
                engine=engine,
                messages_list=[[{"role": "user", "content": "hello"}]],
                sampling_params=SimpleNamespace(),
                chat_template=None,
                chat_template_kwargs={},
                chat_template_content_format="openai",
            )


if __name__ == "__main__":
    unittest.main()
