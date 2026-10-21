from __future__ import annotations

import hashlib
import base64
import itertools
import io
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import logging

from ..base import LLM, LLMRequest, LLMResponse, Capabilities
from proact.core.registry import MODEL_REGISTRY

# Your message/content classes
from ..base import LLMMessage
from ..base import TextPart, ImageURLPart, ImageB64Part, AudioURLPart, AudioB64Part  # noqa: F401

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai"
_API_CALL_COUNTER = itertools.count(1)
_IMAGE_TRANSCODE_CACHE_SIZE = 512


# ---------------------------
# Small utils (same style)
# ---------------------------

def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(inner)) for key, inner in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _stable_kv(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(((key, _freeze_value(value)) for key, value in d.items()), key=lambda kv: kv[0]))


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


def _summarize_responses_input(input_items: list[dict[str, Any]]) -> dict[str, int]:
    text_chars = 0
    image_count = 0
    content_item_count = 0
    for message in input_items:
        content = message.get("content") or []
        if not isinstance(content, list):
            continue
        content_item_count += len(content)
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_text":
                text_chars += len(str(item.get("text") or ""))
            elif item.get("type") == "input_image":
                image_count += 1
    return {
        "messages": len(input_items),
        "content_items": content_item_count,
        "text_chars": text_chars,
        "images": image_count,
    }


class _RateLimiter:
    """
    Simple RPM limiter (single-process).
    rpm=10 -> min_interval=6s between requests.
    """
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


def _transcode_image_b64_part(
    p: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
) -> tuple[str, str]:
    if not image_format or image_format == "source":
        return p.b64, p.mime_type or "image/jpeg"
    if image_format not in {"jpeg", "png", "webp"}:
        raise ValueError(
            "OpenAI image_format must be one of 'source', 'jpeg', 'png', or 'webp'. "
            f"Got {image_format!r}."
        )

    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(p.b64)))
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
    p: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
    cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> tuple[str, str]:
    if cache is None or not image_format or image_format == "source":
        return _transcode_image_b64_part(
            p,
            image_format=image_format,
            image_quality=image_quality,
        )

    key = (p.b64, p.mime_type or "", image_format, image_quality)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    transcoded = _transcode_image_b64_part(
        p,
        image_format=image_format,
        image_quality=image_quality,
    )
    cache[key] = transcoded
    cache.move_to_end(key)
    while len(cache) > _IMAGE_TRANSCODE_CACHE_SIZE:
        cache.popitem(last=False)
    return transcoded


def _content_part_to_responses_item(
    p: Any,
    *,
    image_detail: str | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_transcode_cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> Optional[dict]:
    """
    Map your ContentPart to Responses API content items.

    Responses expects content items like:
      {"type": "input_text", "text": "..."}
      {"type": "input_image", "image_url": "https://..."}  (or data URL with base64)
    """
    # Text
    if isinstance(p, TextPart):
        txt = (p.text or "").strip()
        return {"type": "input_text", "text": txt} if txt else None

    # Image URL
    if isinstance(p, ImageURLPart):
        return {
            "type": "input_image",
            "image_url": str(p.url),
            "detail": image_detail or (p.detail or "auto"),
        }

    # Image base64 => data URL
    if isinstance(p, ImageB64Part):
        b64, mime = _transcode_image_b64_part_cached(
            p,
            image_format=image_format,
            image_quality=image_quality,
            cache=image_transcode_cache,
        )
        data_url = f"data:{mime};base64,{b64}"
        return {
            "type": "input_image",
            "image_url": data_url,
            "detail": image_detail or "auto",
        }

    # Audio (not supported here)
    if isinstance(p, (AudioURLPart, AudioB64Part)):
        raise ValueError("Audio inputs are not supported by this OpenAI adapter.")

    # Unknown: ignore
    return None


def _llm_message_to_responses_input(
    msg: LLMMessage,
    *,
    image_detail: str | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_transcode_cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> dict:
    """
    Convert one LLMMessage into a Responses input item:
      {"role":"user|assistant", "content":[...]}
    """
    role = msg.role

    # Responses input supports roles; for system we’ll move it to `instructions` instead.
    if role not in ("user", "assistant", "system"):
        role = "user"

    if isinstance(msg.content, str):
        content_items = [{"type": "input_text", "text": msg.content}]
    else:
        content_items = []
        for part in msg.content:
            item = _content_part_to_responses_item(
                part,
                image_detail=image_detail,
                image_format=image_format,
                image_quality=image_quality,
                image_transcode_cache=image_transcode_cache,
            )
            if item is not None:
                content_items.append(item)

        # If a message had no usable parts, avoid sending empty content.
        if not content_items:
            content_items = [{"type": "input_text", "text": ""}]

    return {"role": role, "content": content_items}


def _extract_system_instructions(messages: List[LLMMessage]) -> tuple[str, List[LLMMessage]]:
    """
    Responses API has a dedicated `instructions` field (system/developer message). :contentReference[oaicite:3]{index=3}
    We concatenate all system messages into one instructions string and remove them from input.
    """
    sys_chunks: List[str] = []
    kept: List[LLMMessage] = []

    for m in messages:
        if m.role == "system":
            # render system message as plain text (ignore images in system)
            if isinstance(m.content, str):
                if m.content.strip():
                    sys_chunks.append(m.content.strip())
            else:
                # best-effort: keep only text parts
                for part in m.content:
                    if isinstance(part, TextPart) and part.text.strip():
                        sys_chunks.append(part.text.strip())
        else:
            kept.append(m)

    return ("\n\n".join(sys_chunks)).strip(), kept


def _pick_status_code(exc: Exception) -> Optional[int]:
    # openai-python exceptions often expose .status_code; keep this defensive.
    return getattr(exc, "status_code", None)


def _build_llm_response_from_openai(resp_obj: Any) -> LLMResponse:
    """
    Build LLMResponse from Responses API response.

    - resp.output_text is an SDK convenience property aggregating output text. :contentReference[oaicite:4]{index=4}
    - resp.usage includes input_tokens/output_tokens. :contentReference[oaicite:5]{index=5}
    """
    text = getattr(resp_obj, "output_text", None)
    if text is None:
        # fallback: try dict-like
        text = (getattr(resp_obj, "output_text", "") or "")

    usage = getattr(resp_obj, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage is not None else None

    # finish reason is not always a single field on Responses; keep best-effort.
    status = getattr(resp_obj, "status", None)
    finish_reason = "completed" if status == "completed" else status

    raw = resp_obj.model_dump() if hasattr(resp_obj, "model_dump") else (
        resp_obj.to_dict() if hasattr(resp_obj, "to_dict") else resp_obj
    )

    return LLMResponse(
        text=text or "",
        reasoning=None,
        finish_reason=finish_reason,
        usage_tokens_prompt=prompt_tokens,
        usage_tokens_completion=completion_tokens,
        raw=raw,
        ok=True,
        status_code=200,
    )


# ---------------------------
# Adapter
# ---------------------------

@MODEL_REGISTRY.register(PROVIDER_NAME)
class OpenAIResponses(LLM):
    """
    OpenAI Responses API adapter (batch-only complete).

    Supports:
      - text + images (URL or base64/data URL)
      - json_mode via text.format.type="json_object" :contentReference[oaicite:6]{index=6}

    Notes:
      - We group requests by merged sampling params + json_mode to batch in fewer HTTP calls.
      - We do NOT implement streaming here (keep parity with your vllm-local adapter).
    """

    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.RLock()
        self._busy = 0

        # Optional RPM limiter
        self._rpm = getattr(self.config, "rpm_limit", None)
        self._limiter = _RateLimiter(rpm=int(self._rpm)) if self._rpm else None

        # Create OpenAI client lazily to keep import-time failures clean
        self._client = None
        self._client_lock = threading.Lock()
        self._image_transcode_cache: OrderedDict[
            tuple[str, str, str, int | None],
            tuple[str, str],
        ] = OrderedDict()

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

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client

            # Prefer the official SDK
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError(
                    "openai python package not installed. "
                    "Install with `pip install openai`."
                ) from e

            api_key = getattr(self.config, "api_key", None)
            base_url = getattr(self.config, "api_base", None)

            # openai-python uses `base_url=` (also commonly called base_url / base_url in docs)
            kwargs: Dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = str(api_key)
            if base_url:
                kwargs["base_url"] = str(base_url)

            self._client = OpenAI(**kwargs)
            return self._client

    def unload(self, *, force: bool = False) -> None:
        # Nothing persistent to unload (no GPU engine), but keep symmetry.
        with self._lock:
            if self._busy and not force:
                raise RuntimeError("Cannot unload while requests are running (set force=True to override).")
            self._client = None

    def complete(self, reqs: List[LLMRequest]) -> List[LLMResponse]:
        if not reqs:
            return []

        with self._lock:
            self._busy += 1

        try:
            client = self._get_client()
        except Exception as e:
            with self._lock:
                self._busy -= 1
            return [_error_response(str(e), type(e).__name__) for _ in reqs]

        try:
            # Group by (merged params, json_mode, model)
            groups: Dict[str, Dict[str, Any]] = {}

            for idx, rq in enumerate(reqs):
                merged = _merge_params(self.config.params, rq.params)

                # model override allowed (unlike your vllm engine), but keep predictable:
                model = merged.get("model") or self.config.model

                # Only include params we intentionally support for Responses.
                # (Responses supports temperature/top_p/max_output_tokens; see API reference.) :contentReference[oaicite:7]{index=7}
                api_params: Dict[str, Any] = {}
                if "temperature" in merged:
                    api_params["temperature"] = merged["temperature"]
                if "top_p" in merged:
                    api_params["top_p"] = merged["top_p"]
                if isinstance(merged.get("reasoning"), dict):
                    api_params["reasoning"] = merged["reasoning"]
                if isinstance(merged.get("text"), dict):
                    api_params["text"] = merged["text"]

                # Your vLLM adapter uses max_tokens; Responses uses max_output_tokens. :contentReference[oaicite:8]{index=8}
                if "max_output_tokens" in merged:
                    api_params["max_output_tokens"] = merged["max_output_tokens"]
                elif "max_tokens" in merged:
                    api_params["max_output_tokens"] = merged["max_tokens"]

                # Optional seed: docs discuss it in advanced usage, but some SDK typings may lag.
                # We include it only if user explicitly sets it.
                if "seed" in merged:
                    api_params["seed"] = merged["seed"]

                # Build key
                key_tuple = (
                    model,
                    rq.json_mode,
                    _stable_kv(api_params),
                )
                key = hashlib.md5(repr(key_tuple).encode("utf-8")).hexdigest()

                g = groups.setdefault(
                    key,
                    {"indices": [], "requests": [], "model": model, "api_params": api_params, "json_mode": rq.json_mode},
                )
                g["indices"].append(idx)
                g["requests"].append(rq)

            results: List[LLMResponse] = [_error_response("Uninitialized result slot.") for _ in reqs]

            for g in groups.values():
                idxs: List[int] = g["indices"]
                rq_list: List[LLMRequest] = g["requests"]
                model: str = g["model"]
                api_params: Dict[str, Any] = g["api_params"]
                json_mode: bool = g["json_mode"]

                # Call per-request (Responses API does not accept a "batch of conversations" in one request).
                # We still group so we can apply shared throttling, logging, etc.
                for i, rq in zip(idxs, rq_list):
                    if self._limiter:
                        self._limiter.wait()

                    call_id: int | None = None
                    started_at: float | None = None
                    try:
                        call_id = next(_API_CALL_COUNTER)
                        merged = _merge_params(self.config.params, rq.params)
                        image_detail = merged.get("image_detail")
                        if image_detail is not None:
                            image_detail = str(image_detail)
                            if image_detail not in {"low", "high", "auto"}:
                                raise ValueError(
                                    "OpenAI image_detail must be one of 'low', 'high', or 'auto'. "
                                    f"Got {image_detail!r}."
                                )
                        image_format = merged.get("image_format")
                        if image_format is not None:
                            image_format = str(image_format)
                        image_quality = merged.get("image_quality")
                        if image_quality is not None:
                            image_quality = int(image_quality)
                        instructions, non_system_msgs = _extract_system_instructions(rq.messages)
                        provider_prepare_started_at = time.perf_counter()
                        input_items = [
                            _llm_message_to_responses_input(
                                m,
                                image_detail=image_detail,
                                image_format=image_format,
                                image_quality=image_quality,
                                image_transcode_cache=self._image_transcode_cache,
                            )
                            for m in non_system_msgs
                        ]
                        provider_prepare_elapsed_s = time.perf_counter() - provider_prepare_started_at

                        # Structured output control via `text.format`; merge with any profile-level text config.
                        text_cfg = dict(api_params.get("text") or {})
                        text_cfg["format"] = {"type": "json_object" if json_mode else "text"}
                        call_params = dict(api_params)
                        call_params.pop("text", None)

                        input_summary = _summarize_responses_input(input_items)
                        timeout_s = float(self.config.request_timeout_s)
                        logger.info(
                            "OpenAI API call %s started: model=%s json_mode=%s messages=%d content_items=%d "
                            "text_chars=%d images=%d image_detail=%s image_format=%s provider_prepare_s=%.3f "
                            "max_output_tokens=%s timeout_s=%.1f metadata=%s",
                            call_id,
                            model,
                            json_mode,
                            input_summary["messages"],
                            input_summary["content_items"],
                            input_summary["text_chars"],
                            input_summary["images"],
                            image_detail or "auto",
                            image_format or "source",
                            provider_prepare_elapsed_s,
                            call_params.get("max_output_tokens"),
                            timeout_s,
                            _api_log_metadata(rq.metadata),
                        )
                        started_at = time.perf_counter()
                        resp = client.responses.create(
                            model=model,
                            input=input_items,
                            instructions=instructions if instructions else None,
                            text=text_cfg,
                            timeout=timeout_s,
                            **call_params,
                        )
                        results[i] = _build_llm_response_from_openai(resp)
                        elapsed_s = time.perf_counter() - started_at
                        response = results[i]
                        logger.info(
                            "OpenAI API call %s finished: model=%s elapsed_s=%.3f status_code=%s "
                            "finish_reason=%s prompt_tokens=%s completion_tokens=%s output_chars=%d metadata=%s",
                            call_id,
                            model,
                            elapsed_s,
                            response.status_code,
                            response.finish_reason,
                            response.usage_tokens_prompt,
                            response.usage_tokens_completion,
                            len(response.text or ""),
                            _api_log_metadata(rq.metadata),
                        )
                    except Exception as e:
                        elapsed_s = time.perf_counter() - started_at if started_at is not None else None
                        logger.warning(
                            "OpenAI API call %s failed: model=%s elapsed_s=%s status_code=%s error_type=%s "
                            "metadata=%s",
                            call_id if call_id is not None else "not-started",
                            model,
                            f"{elapsed_s:.3f}" if elapsed_s is not None else "not-started",
                            _pick_status_code(e),
                            type(e).__name__,
                            _api_log_metadata(rq.metadata),
                            exc_info=True,
                        )
                        results[i] = _error_response(
                            msg=str(e),
                            exc_type=type(e).__name__,
                            status_code=_pick_status_code(e),
                        )

            return results
        finally:
            with self._lock:
                self._busy -= 1


if __name__ == "__main__":
    import os
    from pathlib import Path
    from ..base import LLMConfig

    # ---- CONFIG ----
    # Make sure you export OPENAI_API_KEY in your shell:
    # export OPENAI_API_KEY=sk-...
    api_key = input("api key... ")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    # Choose a multimodal-capable model
    model_name = "gpt-5.2"  # or another vision-capable model

    config = LLMConfig(
        provider="openai",
        model=model_name,
        api_key=api_key,
        params={
            "temperature": 0.2,
            "max_tokens": 300,
        },
    )

    llm = OpenAIResponses(config)

    # ------------------------------------------------------------
    # 1️⃣ Simple text test
    # ------------------------------------------------------------
    print("\n=== TEXT TEST ===")

    req_text = LLMRequest(
        messages=[
            LLMMessage(role="system", content="You are a concise assistant."),
            LLMMessage(role="user", content="Explain in one sentence what a transformer model is."),
        ]
    )

    resp_text = llm.complete_one(req_text)

    print("OK:", resp_text.ok)
    print("TEXT:", resp_text.text)
    print("TOKENS (prompt/completion):", resp_text.usage_tokens_prompt, "/", resp_text.usage_tokens_completion)

    # ------------------------------------------------------------
    # 2️⃣ Vision test (local image)
    # ------------------------------------------------------------
    print("\n=== VISION TEST ===")

    # Replace with an actual image path on your machine
    image_path = Path("test_image.jpg")

    if image_path.exists():
        img_part = ImageB64Part.from_source(image_path, alt="test image")

        req_vision = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content=[
                        TextPart(text="Describe what you see in this image."),
                        img_part,
                    ],
                )
            ]
        )

        resp_vision = llm.complete_one(req_vision)

        print("OK:", resp_vision.ok)
        print("TEXT:", resp_vision.text)
    else:
        print("Skipping vision test (test_image.jpg not found).")

    # ------------------------------------------------------------
    # 3️⃣ JSON mode test
    # ------------------------------------------------------------
    print("\n=== JSON MODE TEST ===")

    req_json = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Return a JSON object with keys: name (string) and age (integer)."
            )
        ],
        json_mode=True,
    )

    resp_json = llm.complete_one(req_json)

    print("OK:", resp_json.ok)
    print("RAW JSON TEXT:", resp_json.text)

    # ------------------------------------------------------------
    # 4️⃣ Batch test
    # ------------------------------------------------------------
    print("\n=== BATCH TEST ===")

    batch_reqs = [
        LLMRequest(
            messages=[LLMMessage(role="user", content=f"Say hello to user {i}.")]
        )
        for i in range(3)
    ]

    batch_resps = llm.complete(batch_reqs)

    for i, r in enumerate(batch_resps):
        print(f"[{i}] OK={r.ok} | TEXT={r.text}")

    print("\nAll tests finished.")

    import os
    from pathlib import Path
    from PIL import Image
    import numpy as np

    config = LLMConfig(
        provider="openai",
        model=model_name,
        api_key=api_key,
        params={
            "temperature": 0.2,
            "max_tokens": 400,
        },
    )

    llm = OpenAIResponses(config)

    print("\n=== MULTI-IMAGE VISION TEST (384x384) ===")

    # ---- Paths ----
    img1_path = Path("/path/to/proact-benchmark/data/test/Untitled.png")
    img2_path = Path("/path/to/proact-benchmark/data/test/test_picture.jpg")

    if not img1_path.exists() or not img2_path.exists():
        raise RuntimeError("One or both test images not found.")

    # ---- Resize helper ----
    def load_and_resize(path: Path, size=(384, 384)) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BICUBIC)
        return np.array(img)

    img1_arr = load_and_resize(img1_path)
    img2_arr = load_and_resize(img2_path)

    # ---- Convert to ImageB64Part ----
    img1_part = ImageB64Part.from_source(
        img1_arr,
        alt="first test image resized to 384x384"
    )

    img2_part = ImageB64Part.from_source(
        img2_arr,
        alt="second test image resized to 384x384"
    )

    # ---- Build request ----
    req = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content=[
                    TextPart(
                        text="Describe both images together. "
                             "Explain what is visible in each image "
                             "and mention similarities or differences."
                    ),
                    img1_part,
                    img2_part,
                ],
            )
        ]
    )

    resp = llm.complete_one(req)

    print("OK:", resp.ok)
    print("STATUS:", resp.status_code)
    print("PROMPT TOKENS:", resp.usage_tokens_prompt)
    print("COMPLETION TOKENS:", resp.usage_tokens_completion)
    print("\nMODEL OUTPUT:\n")
    print(resp.text)

    print("\n=== DONE ===")
