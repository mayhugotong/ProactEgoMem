from __future__ import annotations
from typing import Dict, Any
from ..base import (
    LLMMessage, TextPart, ImageURLPart, ImageB64Part,
    AudioURLPart, AudioB64Part,
)

def to_openai_chat_message(m: LLMMessage) -> Dict[str, Any]:
    """Convert our message schema to OpenAI-compatible chat content (with vision)."""
    # Plain string → pass as-is
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}

    parts = []
    for p in m.content:
        if isinstance(p, TextPart):
            parts.append({"type": "text", "text": p.text})
        elif isinstance(p, ImageURLPart):
            parts.append({
                "type": "image_url",
                "image_url": {"url": str(p.url), "detail": p.detail or "auto"},
            })
        elif isinstance(p, ImageB64Part):
            data_uri = f"data:{(p.mime_type or 'image/png')};base64,{p.b64}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}})
        elif isinstance(p, (AudioURLPart, AudioB64Part)):
            # Not supported by OpenAI /v1/chat/completions; raise a clear error.
            raise ValueError("Audio inputs are not supported on /v1/chat/completions.")
        else:
            parts.append({"type": "text", "text": "[unsupported content]"})

    return {"role": m.role, "content": parts}
