from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Optional

from proact.core.registry import MODEL_REGISTRY
from proact.models.llm.providers._common import (
    post_json_with_retry,
    transcode_image_b64_part_cached,
)
from proact.models.llm.base import (
    AudioB64Part,
    AudioURLPart,
    Capabilities,
    ImageB64Part,
    ImageURLPart,
    LLM,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TextPart,
)

PROVIDER_NAME = "openrouter"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

_FORWARDED_PARAMS = {
    "frequency_penalty",
    "include_reasoning",
    "logit_bias",
    "max_completion_tokens",
    "max_tokens",
    "min_p",
    "presence_penalty",
    "provider",
    "reasoning",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "user",
    "verbosity",
}


def _error_response(
    message: str,
    error_type: str = "RuntimeError",
    status_code: Optional[int] = None,
) -> LLMResponse:
    return LLMResponse(
        ok=False,
        error=message,
        error_type=error_type,
        status_code=status_code,
    )


class _RateLimiter:
    def __init__(self, *, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self._minimum_interval = 60.0 / float(rpm)
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_request_at
            if elapsed < self._minimum_interval:
                time.sleep(self._minimum_interval - elapsed)
            self._last_request_at = time.time()


def _post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    return post_json_with_retry(
        url=url, headers=headers, payload=payload, timeout_s=timeout_s
    )


def _http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:
        return str(error)


def _content_part(
    part: Any,
    *,
    image_detail: str,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_cache: Any = None,
) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImageURLPart):
        return {
            "type": "image_url",
            "image_url": {
                "url": str(part.url),
                "detail": part.detail or image_detail,
            },
        }
    if isinstance(part, ImageB64Part):
        b64, mime_type = transcode_image_b64_part_cached(
            part,
            image_format=image_format,
            image_quality=image_quality,
            cache=image_cache,
        )
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64}",
                "detail": image_detail,
            },
        }
    if isinstance(part, (AudioURLPart, AudioB64Part)):
        raise ValueError("The OpenRouter adapter does not support audio inputs.")
    raise TypeError(f"Unsupported OpenRouter content part: {type(part)!r}")


def _message_payload(
    message: LLMMessage,
    *,
    image_detail: str,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_cache: Any = None,
) -> dict[str, Any]:
    if isinstance(message.content, str):
        content: str | list[dict[str, Any]] = message.content
    else:
        content = [
            _content_part(
                part,
                image_detail=image_detail,
                image_format=image_format,
                image_quality=image_quality,
                image_cache=image_cache,
            )
            for part in message.content
        ]
    return {"role": message.role, "content": content}


def _response_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _response_reasoning(message: dict[str, Any]) -> Optional[str]:
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    details = message.get("reasoning_details")
    if isinstance(details, list):
        text = "".join(
            item.get("text", "")
            for item in details
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
        return text or None
    return None


def _build_response(payload: dict[str, Any]) -> LLMResponse:
    if payload.get("error"):
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        code = error.get("code") if isinstance(error, dict) else None
        return _error_response(
            message or "OpenRouter returned an error.",
            status_code=code,
        )

    choices = payload.get("choices") or []
    if not choices:
        return _error_response(
            "OpenRouter returned no choices in the response payload.",
            "EmptyResponseError",
        )
    choice = choices[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return LLMResponse(
        text=_response_text(message),
        reasoning=_response_reasoning(message),
        finish_reason=choice.get("finish_reason"),
        usage_tokens_prompt=usage.get("prompt_tokens"),
        usage_tokens_completion=usage.get("completion_tokens"),
        raw=payload,
        ok=True,
        status_code=200,
    )


@MODEL_REGISTRY.register(PROVIDER_NAME)
class OpenRouterChatCompletions(LLM):
    def __init__(self, config):
        super().__init__(config)
        rpm = getattr(self.config, "rpm_limit", None)
        self._limiter = _RateLimiter(rpm=int(rpm)) if rpm else None
        self._image_transcode_cache: OrderedDict = OrderedDict()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_streaming=False,
            supports_json_mode=True,
            supports_system=True,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            supports_thinking=True,
        )

    def complete(self, reqs: list[LLMRequest]) -> list[LLMResponse]:
        if not reqs:
            return []
        if not self.config.api_key:
            return [
                _error_response(
                    "OpenRouter provider requires api_key in the selected LLM profile.",
                    "ValueError",
                )
                for _ in reqs
            ]

        api_base = str(self.config.api_base or DEFAULT_API_BASE).rstrip("/")
        extra = dict(self.config.extra or {})
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if extra.get("http_referer"):
            headers["HTTP-Referer"] = str(extra["http_referer"])
        if extra.get("app_title"):
            headers["X-OpenRouter-Title"] = str(extra["app_title"])

        results: list[LLMResponse] = []
        for request in reqs:
            if self._limiter:
                self._limiter.wait()
            try:
                merged = dict(self.config.params or {})
                merged.update(request.params or {})
                image_detail = str(merged.get("image_detail", "auto"))
                image_format = merged.get("image_format")
                image_quality = merged.get("image_quality")
                model = str(merged.get("model") or self.config.model)
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        _message_payload(
                            message,
                            image_detail=image_detail,
                            image_format=image_format,
                            image_quality=(
                                int(image_quality) if image_quality is not None else None
                            ),
                            image_cache=self._image_transcode_cache,
                        )
                        for message in request.messages
                    ],
                }
                for key in _FORWARDED_PARAMS:
                    if key in merged:
                        payload[key] = merged[key]
                if "max_output_tokens" in merged and "max_tokens" not in payload:
                    payload["max_tokens"] = merged["max_output_tokens"]
                if request.json_mode:
                    payload["response_format"] = {"type": "json_object"}

                response_payload = _post_json(
                    url=f"{api_base}/chat/completions",
                    headers=headers,
                    payload=payload,
                    timeout_s=float(self.config.request_timeout_s),
                )
                results.append(_build_response(response_payload))
            except urllib.error.HTTPError as error:
                results.append(
                    _error_response(
                        _http_error_body(error),
                        type(error).__name__,
                        status_code=error.code,
                    )
                )
            except urllib.error.URLError as error:
                results.append(_error_response(str(error), type(error).__name__))
            except Exception as error:
                results.append(_error_response(str(error), type(error).__name__))
        return results
