from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoWindowRequest:
    input_video_path: str
    prompt_text: str
    state_text: str
    frames: list[Any]
    frame_timestamps: list[float]
    step_index: int
    timestamp_sec: float
    input_mode: str
    input_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    adapter_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoLLMResponse:
    response_text: str = ""
    response_ok: bool = True
    response_error: str | None = None
    response_error_type: str | None = None
    response_status_code: int | None = None
    response_finish_reason: str | None = None
    response_reasoning: str | None = None
    response_raw: Any = None
    raw_model_output: dict[str, Any] = field(default_factory=dict)
    request_messages: list[dict[str, Any]] = field(default_factory=list)


def build_interleaved_video_window_text(
    *,
    prompt_text: str,
    state_text: str,
    frame_timestamps: list[float],
    image_placeholder: str = "[IMAGE]",
) -> str:
    """Render the frame window as interleaved image/timestamp lines.

    Shared by the video-LLM backend and adapters so every model family sees the
    same frame-label wording as the stateless prompt builder
    ("Frame #N at X.XXX seconds")."""
    frame_lines = "\n".join(
        f"{image_placeholder}\nFrame #{index + 1} at {timestamp:.3f} seconds"
        for index, timestamp in enumerate(frame_timestamps)
    )
    parts = [prompt_text]
    if state_text:
        parts.append(state_text)
    parts.append(f"The window of video frames\n{frame_lines}")
    return "\n\n".join(parts)


class VideoLLMAdapter(ABC):
    @abstractmethod
    def complete_window(self, request: VideoWindowRequest) -> VideoLLMResponse:
        raise NotImplementedError
