from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from proact.core.registry import MODEL_REGISTRY
from proact.models.llm.providers._common import post_json_with_retry
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

logger = logging.getLogger(__name__)

PROVIDER_NAME = "gemini"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(inner)) for key, inner in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _stable_kv(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, _freeze_value(value)) for key, value in d.items()))


def _merge_params(global_params: Dict[str, Any], req_params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_params or {})
    merged.update(req_params or {})
    return merged


def _error_response(msg: str, exc_type: str = "RuntimeError", status_code: Optional[int] = None) -> LLMResponse:
    return LLMResponse(
        ok=False,
        error=msg,
        error_type=exc_type,
        status_code=status_code,
    )


class _RateLimiter:
    def __init__(self, *, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self._min_interval = 60.0 / float(rpm)
        self._last_ts = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            dt = now - self._last_ts
            if dt < self._min_interval:
                time.sleep(self._min_interval - dt)
            self._last_ts = time.time()


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


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


def _extract_system_instructions(messages: List[LLMMessage]) -> tuple[str, List[LLMMessage]]:
    system_chunks: list[str] = []
    kept: list[LLMMessage] = []
    for message in messages:
        if message.role != "system":
            kept.append(message)
            continue
        if isinstance(message.content, str):
            text = message.content.strip()
            if text:
                system_chunks.append(text)
            continue
        for part in message.content:
            if isinstance(part, TextPart) and part.text.strip():
                system_chunks.append(part.text.strip())
    return ("\n\n".join(system_chunks)).strip(), kept


def _camelize_key(key: str) -> str:
    if "_" not in key:
        return key
    head, *tail = key.split("_")
    return head + "".join(token.capitalize() for token in tail)


def _camelize_dict(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        normalized_key = _camelize_key(key)
        if isinstance(value, dict):
            result[normalized_key] = _camelize_dict(value)
        elif isinstance(value, list):
            result[normalized_key] = [
                _camelize_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[normalized_key] = value
    return result


def _content_part_to_gemini_part(part: Any) -> Optional[dict[str, Any]]:
    if isinstance(part, TextPart):
        text = (part.text or "").strip()
        return {"text": text} if text else None

    if isinstance(part, ImageB64Part):
        return {
            "inline_data": {
                "mime_type": part.mime_type or "image/png",
                "data": part.b64,
            }
        }

    if isinstance(part, ImageURLPart):
        raise ValueError("Gemini adapter does not support ImageURLPart. Use ImageB64Part inputs instead.")

    if isinstance(part, (AudioURLPart, AudioB64Part)):
        raise ValueError("Audio inputs are not supported by this Gemini adapter.")

    return None


def _llm_message_to_gemini_content(message: LLMMessage) -> dict[str, Any]:
    role = "model" if message.role == "assistant" else "user"
    if isinstance(message.content, str):
        return {"role": role, "parts": [{"text": message.content}]}

    parts: list[dict[str, Any]] = []
    for part in message.content:
        item = _content_part_to_gemini_part(part)
        if item is not None:
            parts.append(item)
    if not parts:
        parts = [{"text": ""}]
    return {"role": role, "parts": parts}


def _build_llm_response_from_gemini(response_payload: dict[str, Any]) -> LLMResponse:
    candidates = response_payload.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text_fragments = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    usage = response_payload.get("usageMetadata") or {}
    return LLMResponse(
        text="".join(text_fragments),
        reasoning=None,
        finish_reason=candidate.get("finishReason"),
        usage_tokens_prompt=usage.get("promptTokenCount"),
        usage_tokens_completion=usage.get("candidatesTokenCount"),
        raw=response_payload,
        ok=True,
        status_code=200,
    )


@MODEL_REGISTRY.register(PROVIDER_NAME)
class GeminiGenerateContent(LLM):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.RLock()
        self._busy = 0
        self._rpm = getattr(self.config, "rpm_limit", None)
        self._limiter = _RateLimiter(rpm=int(self._rpm)) if self._rpm else None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_streaming=False,
            supports_json_mode=True,
            supports_system=True,
            supports_tools=False,
            supports_vision=True,
            supports_audio=False,
            supports_thinking=True,
        )

    def complete(self, reqs: List[LLMRequest]) -> List[LLMResponse]:
        if not reqs:
            return []

        with self._lock:
            self._busy += 1

        try:
            groups: Dict[str, Dict[str, Any]] = {}
            for idx, request in enumerate(reqs):
                merged = _merge_params(self.config.params, request.params)
                model = merged.get("model") or self.config.model

                generation_config: dict[str, Any] = {}
                if "temperature" in merged:
                    generation_config["temperature"] = merged["temperature"]
                if "top_p" in merged:
                    generation_config["topP"] = merged["top_p"]
                if "top_k" in merged:
                    generation_config["topK"] = merged["top_k"]
                if "max_output_tokens" in merged:
                    generation_config["maxOutputTokens"] = merged["max_output_tokens"]
                elif "max_tokens" in merged:
                    generation_config["maxOutputTokens"] = merged["max_tokens"]
                if request.json_mode:
                    generation_config["responseMimeType"] = "application/json"
                if isinstance(merged.get("thinking_config"), dict):
                    generation_config["thinkingConfig"] = _camelize_dict(merged["thinking_config"])

                key_tuple = (model, request.json_mode, _stable_kv(generation_config))
                key = hashlib.md5(repr(key_tuple).encode("utf-8")).hexdigest()
                group = groups.setdefault(
                    key,
                    {
                        "indices": [],
                        "requests": [],
                        "model": model,
                        "generation_config": generation_config,
                    },
                )
                group["indices"].append(idx)
                group["requests"].append(request)

            results: List[LLMResponse] = [_error_response("Uninitialized result slot.") for _ in reqs]
            api_base = str(self.config.api_base or DEFAULT_API_BASE).rstrip("/")
            if not self.config.api_key:
                return [
                    _error_response(
                        "Gemini provider requires api_key in the selected LLM profile.",
                        "ValueError",
                    )
                    for _ in reqs
                ]

            for group in groups.values():
                model = group["model"]
                generation_config = dict(group["generation_config"])
                for index, request in zip(group["indices"], group["requests"]):
                    if self._limiter:
                        self._limiter.wait()
                    try:
                        system_instruction, non_system_messages = _extract_system_instructions(request.messages)
                        payload: dict[str, Any] = {
                            "contents": [
                                _llm_message_to_gemini_content(message)
                                for message in non_system_messages
                            ]
                        }
                        if system_instruction:
                            payload["system_instruction"] = {
                                "parts": [{"text": system_instruction}]
                            }
                        if generation_config:
                            payload["generationConfig"] = generation_config

                        response_payload = _post_json(
                            url=f"{api_base}/models/{model}:generateContent",
                            headers={
                                "Content-Type": "application/json",
                                "x-goog-api-key": str(self.config.api_key),
                            },
                            payload=payload,
                            timeout_s=float(self.config.request_timeout_s),
                        )
                        results[index] = _build_llm_response_from_gemini(response_payload)
                    except urllib.error.HTTPError as exc:
                        results[index] = _error_response(
                            _http_error_body(exc),
                            type(exc).__name__,
                            status_code=exc.code,
                        )
                    except urllib.error.URLError as exc:
                        results[index] = _error_response(str(exc), type(exc).__name__)
                    except Exception as exc:
                        results[index] = _error_response(str(exc), type(exc).__name__)
            return results
        finally:
            with self._lock:
                self._busy -= 1
