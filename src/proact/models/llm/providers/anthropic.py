from __future__ import annotations

import hashlib
import base64
import itertools
import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
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

PROVIDER_NAME = "anthropic"
DEFAULT_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
logger = logging.getLogger(__name__)
_API_CALL_COUNTER = itertools.count(1)
_IMAGE_TRANSCODE_CACHE_SIZE = 512


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


def _api_log_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    selected_keys = (
        "parent_video_uid",
        "match_id",
        "step_index",
        "timestamp_sec",
        "evaluation_context",
        "input_mode",
        "sampled_frame_count",
    )
    return {key: metadata[key] for key in selected_keys if key in metadata}


def _summarize_anthropic_messages(messages: list[dict[str, Any]]) -> dict[str, int]:
    text_chars = 0
    image_count = 0
    content_item_count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content_item_count += 1
            text_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        content_item_count += len(content)
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_chars += len(str(item.get("text") or ""))
            elif item.get("type") == "image":
                image_count += 1
    return {
        "messages": len(messages),
        "content_items": content_item_count,
        "text_chars": text_chars,
        "images": image_count,
    }


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


def _transcode_image_b64_part(
    part: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
) -> tuple[str, str]:
    if not image_format or image_format == "source":
        return part.b64, part.mime_type or "image/png"
    if image_format not in {"jpeg", "png", "webp"}:
        raise ValueError(
            "Anthropic image_format must be one of 'source', 'jpeg', 'png', or 'webp'. "
            f"Got {image_format!r}."
        )

    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(part.b64)))
    pil_format = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}[image_format]
    mime_type = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[image_format]
    if pil_format in {"JPEG", "WEBP"}:
        image = image.convert("RGB")

    save_kwargs: dict[str, Any] = {}
    if image_quality is not None and pil_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = int(image_quality)
        if pil_format == "JPEG":
            save_kwargs["optimize"] = True

    buffer = io.BytesIO()
    image.save(buffer, format=pil_format, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), mime_type


def _transcode_image_b64_part_cached(
    part: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
    cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> tuple[str, str]:
    if cache is None or not image_format or image_format == "source":
        return _transcode_image_b64_part(
            part,
            image_format=image_format,
            image_quality=image_quality,
        )

    key = (part.b64, part.mime_type or "", image_format, image_quality)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    transcoded = _transcode_image_b64_part(
        part,
        image_format=image_format,
        image_quality=image_quality,
    )
    cache[key] = transcoded
    cache.move_to_end(key)
    while len(cache) > _IMAGE_TRANSCODE_CACHE_SIZE:
        cache.popitem(last=False)
    return transcoded


def _content_part_to_anthropic_block(
    part: Any,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_transcode_cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> Optional[dict[str, Any]]:
    if isinstance(part, TextPart):
        text = (part.text or "").strip()
        return {"type": "text", "text": text} if text else None

    if isinstance(part, ImageB64Part):
        b64, media_type = _transcode_image_b64_part_cached(
            part,
            image_format=image_format,
            image_quality=image_quality,
            cache=image_transcode_cache,
        )
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }

    if isinstance(part, ImageURLPart):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": str(part.url),
            },
        }

    if isinstance(part, (AudioURLPart, AudioB64Part)):
        raise ValueError("Audio inputs are not supported by this Anthropic adapter.")

    return None


def _llm_message_to_anthropic_message(
    message: LLMMessage,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_transcode_cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> dict[str, Any]:
    role = "assistant" if message.role == "assistant" else "user"
    if isinstance(message.content, str):
        return {"role": role, "content": message.content}

    blocks: list[dict[str, Any]] = []
    for part in message.content:
        block = _content_part_to_anthropic_block(
            part,
            image_format=image_format,
            image_quality=image_quality,
            image_transcode_cache=image_transcode_cache,
        )
        if block is not None:
            blocks.append(block)
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return {"role": role, "content": blocks}


def _build_llm_response_from_anthropic(response_payload: dict[str, Any]) -> LLMResponse:
    content_blocks = response_payload.get("content") or []
    text_fragments = [
        block.get("text", "")
        for block in content_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    usage = response_payload.get("usage") or {}
    return LLMResponse(
        text="".join(text_fragments),
        reasoning=None,
        finish_reason=response_payload.get("stop_reason"),
        usage_tokens_prompt=usage.get("input_tokens"),
        usage_tokens_completion=usage.get("output_tokens"),
        raw=response_payload,
        ok=True,
        status_code=200,
    )


@MODEL_REGISTRY.register(PROVIDER_NAME)
class AnthropicMessages(LLM):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.RLock()
        self._busy = 0
        self._rpm = getattr(self.config, "rpm_limit", None)
        self._limiter = _RateLimiter(rpm=int(self._rpm)) if self._rpm else None
        self._image_transcode_cache: OrderedDict[
            tuple[str, str, str, int | None],
            tuple[str, str],
        ] = OrderedDict()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_streaming=False,
            supports_json_mode=False,
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

                api_params: dict[str, Any] = {}
                if "temperature" in merged:
                    api_params["temperature"] = merged["temperature"]
                if "top_p" in merged:
                    api_params["top_p"] = merged["top_p"]
                if "top_k" in merged:
                    api_params["top_k"] = merged["top_k"]
                if "stop_sequences" in merged:
                    api_params["stop_sequences"] = merged["stop_sequences"]
                if "max_tokens" in merged:
                    api_params["max_tokens"] = merged["max_tokens"]
                elif "max_output_tokens" in merged:
                    api_params["max_tokens"] = merged["max_output_tokens"]
                if isinstance(merged.get("thinking"), dict):
                    api_params["thinking"] = merged["thinking"]

                key_tuple = (model, request.json_mode, _stable_kv(api_params))
                key = hashlib.md5(repr(key_tuple).encode("utf-8")).hexdigest()
                group = groups.setdefault(
                    key,
                    {"indices": [], "requests": [], "model": model, "api_params": api_params},
                )
                group["indices"].append(idx)
                group["requests"].append(request)

            results: List[LLMResponse] = [_error_response("Uninitialized result slot.") for _ in reqs]
            api_base = str(self.config.api_base or DEFAULT_API_BASE).rstrip("/")
            if not self.config.api_key:
                return [
                    _error_response(
                        "Anthropic provider requires api_key in the selected LLM profile.",
                        "ValueError",
                    )
                    for _ in reqs
                ]

            anthropic_version = str(self.config.extra.get("anthropic_version", DEFAULT_ANTHROPIC_VERSION))
            anthropic_beta = self.config.extra.get("anthropic_beta")

            for group in groups.values():
                model = group["model"]
                api_params = dict(group["api_params"])
                for index, request in zip(group["indices"], group["requests"]):
                    if self._limiter:
                        self._limiter.wait()
                    call_id: int | None = None
                    started_at: float | None = None
                    try:
                        call_id = next(_API_CALL_COUNTER)
                        system_instruction, non_system_messages = _extract_system_instructions(request.messages)
                        merged = _merge_params(self.config.params, request.params)
                        image_format = merged.get("image_format")
                        if image_format is not None:
                            image_format = str(image_format)
                        image_quality = merged.get("image_quality")
                        if image_quality is not None:
                            image_quality = int(image_quality)
                        provider_prepare_started_at = time.perf_counter()
                        messages = [
                            _llm_message_to_anthropic_message(
                                message,
                                image_format=image_format,
                                image_quality=image_quality,
                                image_transcode_cache=self._image_transcode_cache,
                            )
                            for message in non_system_messages
                        ]
                        provider_prepare_elapsed_s = time.perf_counter() - provider_prepare_started_at

                        payload: dict[str, Any] = {
                            "model": model,
                            "messages": messages,
                        }
                        payload.update(api_params)
                        payload.setdefault("max_tokens", 1024)
                        if system_instruction:
                            payload["system"] = system_instruction

                        headers = {
                            "Content-Type": "application/json",
                            "x-api-key": str(self.config.api_key),
                            "anthropic-version": anthropic_version,
                        }
                        if anthropic_beta:
                            headers["anthropic-beta"] = (
                                ",".join(anthropic_beta)
                                if isinstance(anthropic_beta, list)
                                else str(anthropic_beta)
                            )

                        message_summary = _summarize_anthropic_messages(payload["messages"])
                        timeout_s = float(self.config.request_timeout_s)
                        logger.info(
                            "Anthropic API call %s started: model=%s json_mode=%s messages=%d content_items=%d "
                            "text_chars=%d images=%d image_format=%s provider_prepare_s=%.3f "
                            "max_tokens=%s timeout_s=%.1f metadata=%s",
                            call_id,
                            model,
                            request.json_mode,
                            message_summary["messages"],
                            message_summary["content_items"],
                            message_summary["text_chars"],
                            message_summary["images"],
                            image_format or "source",
                            provider_prepare_elapsed_s,
                            payload.get("max_tokens"),
                            timeout_s,
                            _api_log_metadata(request.metadata),
                        )
                        started_at = time.perf_counter()
                        response_payload = _post_json(
                            url=f"{api_base}/v1/messages",
                            headers=headers,
                            payload=payload,
                            timeout_s=timeout_s,
                        )
                        results[index] = _build_llm_response_from_anthropic(response_payload)
                        elapsed_s = time.perf_counter() - started_at
                        response = results[index]
                        logger.info(
                            "Anthropic API call %s finished: model=%s elapsed_s=%.3f status_code=%s "
                            "finish_reason=%s prompt_tokens=%s completion_tokens=%s output_chars=%d metadata=%s",
                            call_id,
                            model,
                            elapsed_s,
                            response.status_code,
                            response.finish_reason,
                            response.usage_tokens_prompt,
                            response.usage_tokens_completion,
                            len(response.text or ""),
                            _api_log_metadata(request.metadata),
                        )
                    except urllib.error.HTTPError as exc:
                        elapsed_s = time.perf_counter() - started_at if started_at is not None else None
                        logger.warning(
                            "Anthropic API call %s failed: model=%s elapsed_s=%s status_code=%s error_type=%s "
                            "metadata=%s",
                            call_id if call_id is not None else "not-started",
                            model,
                            f"{elapsed_s:.3f}" if elapsed_s is not None else "not-started",
                            exc.code,
                            type(exc).__name__,
                            _api_log_metadata(request.metadata),
                            exc_info=True,
                        )
                        results[index] = _error_response(
                            _http_error_body(exc),
                            type(exc).__name__,
                            status_code=exc.code,
                        )
                    except urllib.error.URLError as exc:
                        elapsed_s = time.perf_counter() - started_at if started_at is not None else None
                        logger.warning(
                            "Anthropic API call %s failed: model=%s elapsed_s=%s status_code=%s error_type=%s "
                            "metadata=%s",
                            call_id if call_id is not None else "not-started",
                            model,
                            f"{elapsed_s:.3f}" if elapsed_s is not None else "not-started",
                            None,
                            type(exc).__name__,
                            _api_log_metadata(request.metadata),
                            exc_info=True,
                        )
                        results[index] = _error_response(str(exc), type(exc).__name__)
                    except Exception as exc:
                        elapsed_s = time.perf_counter() - started_at if started_at is not None else None
                        logger.warning(
                            "Anthropic API call %s failed: model=%s elapsed_s=%s status_code=%s error_type=%s "
                            "metadata=%s",
                            call_id if call_id is not None else "not-started",
                            model,
                            f"{elapsed_s:.3f}" if elapsed_s is not None else "not-started",
                            None,
                            type(exc).__name__,
                            _api_log_metadata(request.metadata),
                            exc_info=True,
                        )
                        results[index] = _error_response(str(exc), type(exc).__name__)
            return results
        finally:
            with self._lock:
                self._busy -= 1
