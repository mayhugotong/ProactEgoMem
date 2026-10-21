from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, AsyncIterator, Union
from pydantic import BaseModel, Field, HttpUrl
import numpy as np
from pathlib import Path

Role = Literal["system", "user", "assistant"]

# ---------- Multimodal content parts ----------
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageURLPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    url: HttpUrl
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    alt: Optional[str] = None  # optional human hint

class ImageB64Part(BaseModel):
    type: Literal["image_b64"] = "image_b64"
    b64: str = Field(repr=False)  
    mime_type: Optional[str] = None
    alt: Optional[str] = None

    def __str__(self):
        return "[IMAGE]"

    @classmethod
    def from_source(
        cls,
        src,
        *,
        alt: str | None = None,
        mime_type: str | None = None,
        assume_bgr: bool = False,
    ) -> "ImageB64Part":
        """
        Create an ImageB64Part from various sources:
        - path-like (str/Path) → load file
        - numpy.ndarray        → encode directly
        - existing ImageB64Part → return as-is / clone
        """
        

        # Already an ImageB64Part
        if isinstance(src, ImageB64Part):
            # optionally override alt/mime_type
            data = src.model_copy(update={
                "alt": alt if alt is not None else src.alt,
                "mime_type": mime_type if mime_type is not None else src.mime_type,
            })
            return data

        # Path-like → use your file helper
        if isinstance(src, (str, bytes, Path)):
            return image_file_to_b64_part(src, alt=alt)

        # Numpy array → use your ndarray helper
        if isinstance(src, np.ndarray):
            mt = mime_type or "image/png"
            return ndarray_to_b64_part(src, mime=mt, alt=alt, assume_bgr=assume_bgr)

        raise TypeError(
            f"Unsupported source type for ImageB64Part: {type(src)!r}. "
            "Expected path-like, numpy.ndarray, or ImageB64Part."
        )

# Helper for ImageB64Part
import base64, io, mimetypes
import numpy as np
from PIL import Image

def image_file_to_b64_part(path: str | bytes | Path, *, alt: str | None = None) -> ImageB64Part:
    """
    Read a local image file and return an ImageB64Part with a best-effort MIME type.
    No server-side download; works offline.
    """
    p = Path(path)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")

    # Guess MIME from extension; default to image/jpeg if unknown
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        # Common fallback for .jpg/.jpeg-like content
        mime = "image/jpeg"

    return ImageB64Part(b64=b64, mime_type=mime, alt=alt)

def ndarray_to_b64_part(arr: np.ndarray, *, mime: str = "image/png",
                        alt: str | None = None, assume_bgr: bool = False) -> ImageB64Part:
    """
    Convert a numpy image array (H,W[,C]) to ImageB64Part.
    - dtype: uint8 preferred. If float, expects 0..1; will scale to 0..255.
    - channels: 2D/gray, 3 (RGB/BGR), or 4 (RGBA).
    - If the source was read by OpenCV, set assume_bgr=True.
    """
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    a = arr
    # dtype handling
    if a.dtype == np.float32 or a.dtype == np.float64:
        a_min, a_max = float(a.min()), float(a.max())
        if a_min < 0.0 or a_max > 1.0:
            raise ValueError(f"Float image expected in [0,1], got range [{a_min},{a_max}]")
        a = (a * 255.0).round().astype(np.uint8)
    elif a.dtype != np.uint8:
        # Best-effort cast for other integer types
        a = a.astype(np.uint8)

    # color order
    if a.ndim == 3 and a.shape[2] == 3 and assume_bgr:
        a = a[..., ::-1]  # BGR -> RGB

    # PIL mode conversion
    if a.ndim == 2:
        img = Image.fromarray(a, mode="L")
    elif a.shape[2] == 1:
        img = Image.fromarray(a.squeeze(-1), mode="L")
    elif a.shape[2] == 3:
        img = Image.fromarray(a, mode="RGB")
    elif a.shape[2] == 4:
        img = Image.fromarray(a, mode="RGBA")
    else:
        raise ValueError(f"Unsupported channel count: {a.shape[2]}")

    # JPEG can't store alpha—drop it if needed
    if mime == "image/jpeg" and img.mode == "RGBA":
        img = img.convert("RGB")

    buf = io.BytesIO()
    pil_format = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/webp": "WEBP",
        "image/bmp": "BMP",
        "image/tiff": "TIFF",
    }.get(mime, "PNG")

    save_kwargs = {}
    if pil_format == "JPEG":
        save_kwargs.update({"quality": 90, "optimize": True})

    img.save(buf, format=pil_format, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return ImageB64Part(b64=b64, mime_type=mime, alt=alt)


class AudioURLPart(BaseModel):
    type: Literal["audio_url"] = "audio_url"
    url: HttpUrl
    mime_type: Optional[str] = None
    alt: Optional[str] = None

class AudioB64Part(BaseModel):
    type: Literal["audio_b64"] = "audio_b64"
    b64: str
    mime_type: Optional[str] = None
    alt: Optional[str] = None

ContentPart = Union[
    TextPart,
    ImageURLPart, ImageB64Part,
    AudioURLPart, AudioB64Part,
]

from typing import Any

class LLMMessage(BaseModel):
    role: Role
    content: Union[str, List[ContentPart]]

    def to_string(
        self,
        *,
        include_role: bool = False,
        keep_trailing_newline: bool = False,
        include_alt: bool = True,
        include_urls: bool = False,
    ) -> str:
        """
        Render this message into a prompt-friendly string.

        - TextPart => its text
        - Image*   => [IMAGE] (+ optional alt / url)
        - Audio*   => [AUDIO] (+ optional alt / url)
        - str      => returned as-is
        """
        def render_part(p: Any) -> str:
            # Text
            if isinstance(p, TextPart):
                return p.text

            # Images
            if isinstance(p, ImageB64Part):
                if include_alt and p.alt:
                    return f"[IMAGE] {p.alt}"
                return "[IMAGE]"

            if isinstance(p, ImageURLPart):
                pieces = ["[IMAGE]"]
                if include_alt and p.alt:
                    pieces.append(p.alt)
                if include_urls:
                    pieces.append(str(p.url))
                return " ".join(pieces)

            # Audio
            if isinstance(p, AudioB64Part):
                if include_alt and p.alt:
                    return f"[AUDIO] {p.alt}"
                return "[AUDIO]"

            if isinstance(p, AudioURLPart):
                pieces = ["[AUDIO]"]
                if include_alt and p.alt:
                    pieces.append(p.alt)
                if include_urls:
                    pieces.append(str(p.url))
                return " ".join(pieces)

            # Fallbacks (don’t crash)
            if hasattr(p, "text"):
                return str(getattr(p, "text"))
            return str(p)

        out_parts: list[str] = []
        if include_role:
            out_parts.append(f"{self.role}")

        if isinstance(self.content, str):
            out_parts.append(self.content)
        else:
            for part in self.content:
                s = render_part(part)
                if s is None:
                    continue
                s = s.strip()
                if s:
                    out_parts.append(s)

        res = "\n".join(out_parts)
        if keep_trailing_newline:
            return res + "\n"
        return res


class LLMToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class LLMRequest(BaseModel):
    messages: List[LLMMessage]
    json_mode: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    text: Optional[str] = ""
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage_tokens_prompt: Optional[int] = None
    usage_tokens_completion: Optional[int] = None
    raw: Any = Field(default_factory=dict)
    # Additions for requests over API
    ok: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None
    status_code: Optional[int] = None

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_base: Optional[HttpUrl] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0
    tps_limit: Optional[float] = None
    rpm_limit: Optional[int] = None
    cache_enabled: bool = False

    # ↓↓↓ keep sampling/generation separate
    params: Dict[str, Any] = Field(default_factory=dict)

    # ↓↓↓ non-sampling request/render stuff (e.g., chat_template_kwargs)
    extra: Dict[str, Any] = Field(default_factory=dict)

    # ↓↓↓ NEW: engine constructor kwargs (vLLM init-time)
    engine: Dict[str, Any] = Field(default_factory=dict)

    
class Capabilities(BaseModel):
    supports_streaming: bool = True
    supports_json_mode: bool = True
    supports_system: bool = True
    supports_thinking: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False

class LLM(ABC):
    """
    Provider-agnostic LLM interface. Implementations must be thin adapters.

    Core contract (batch-only):

      - complete(reqs)  -> list of responses, len(reqs) == len(responses)

    Convenience single-request helpers are provided on this base class.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities()

    # ---------- core batch API ----------

    @abstractmethod
    def complete(
        self,
        reqs: List[LLMRequest],
    ) -> List[LLMResponse]:
        """
        Unified completion API (batch-only).

        - Input: list of LLMRequest
        - Output: list of LLMResponse with the same length/order.
        """
        ...

    # ---------- single-request convenience wrappers ----------

    def complete_one(self, req: LLMRequest) -> LLMResponse:
        """
        Convenience wrapper around batch `complete` for a single request.
        """
        res_list = self.complete([req])
        # Defensive: if an implementation ever returns wrong length,
        # still avoid blowing up call sites.
        return res_list[0] if res_list else LLMResponse(
            ok=False,
            error="Complete returned no responses for single request.",
            error_type="RuntimeError",
        )

    async def acomplete(
        self,
        reqs: List[LLMRequest],
    ) -> List[LLMResponse]:
        """
        Async batch completion. Default: run sync in a thread-pool executor.
        Providers can override this with a true async implementation.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.complete, reqs)

    async def acomplete_one(self, req: LLMRequest) -> LLMResponse:
        res_list = await self.acomplete([req])
        return res_list[0] if res_list else LLMResponse(
            ok=False,
            error="Acomplete returned no responses for single request.",
            error_type="RuntimeError",
        )

    # ---------- streaming helpers (still single-request) ----------

    def stream(self, req: LLMRequest) -> Iterable[str]:
        """
        Optional token stream. Default implementation folds to non-streaming:
        calls `complete_one` and yields the final text as a single chunk.
        """
        resp = self.complete_one(req)
        yield resp.text or ""

    async def astream(self, req: LLMRequest) -> AsyncIterator[str]:
        for chunk in self.stream(req):
            yield chunk