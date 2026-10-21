"""Shared helpers for the urllib-based LLM providers.

Single source for HTTP POST with bounded retry/backoff and image transcoding,
so transient 429/5xx handling and image fidelity are identical across
providers instead of drifting per file.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any

from proact.models.llm.base import ImageB64Part

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0

IMAGE_TRANSCODE_CACHE_SIZE = 512


def post_json_with_retry(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> dict[str, Any]:
    """POST JSON with bounded exponential backoff on transient failures.

    Retries on retryable HTTP status codes, connection errors, and timeouts.
    Non-retryable HTTP errors (4xx other than 408/409/429) raise immediately.
    After the final attempt the last error propagates to the caller, which is
    expected to turn it into an ok=False response (and, in the evaluation
    pipeline, a loud clip failure).
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if attempt < retries and error.code in RETRYABLE_STATUS_CODES:
                last_error = error
            else:
                raise
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < retries:
                last_error = error
            else:
                raise
        delay = backoff_s * (2**attempt)
        logger.warning(
            "Provider request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
            url,
            attempt + 1,
            retries + 1,
            last_error,
            delay,
        )
        time.sleep(delay)
    raise RuntimeError(f"Provider request to {url} failed after {retries + 1} attempts: {last_error}")


def transcode_image_b64_part(
    p: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
) -> tuple[str, str]:
    """Re-encode an image part; returns (b64, mime_type).

    image_format 'source'/None passes the image through unchanged.
    """
    if not image_format or image_format == "source":
        return p.b64, p.mime_type or "image/jpeg"
    if image_format not in {"jpeg", "png", "webp"}:
        raise ValueError(
            "image_format must be one of 'source', 'jpeg', 'png', or 'webp'. "
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


def transcode_image_b64_part_cached(
    p: ImageB64Part,
    *,
    image_format: str | None = None,
    image_quality: int | None = None,
    cache: OrderedDict[tuple[str, str, str, int | None], tuple[str, str]] | None = None,
) -> tuple[str, str]:
    if cache is None or not image_format or image_format == "source":
        return transcode_image_b64_part(
            p, image_format=image_format, image_quality=image_quality
        )

    key = (p.b64, p.mime_type or "", image_format, image_quality)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    transcoded = transcode_image_b64_part(
        p, image_format=image_format, image_quality=image_quality
    )
    cache[key] = transcoded
    cache.move_to_end(key)
    while len(cache) > IMAGE_TRANSCODE_CACHE_SIZE:
        cache.popitem(last=False)
    return transcoded
