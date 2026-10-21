from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingOutputEvent:
    timestamp_sec: float
    response_text: str
    sampled_frame_times: list[float] = field(default_factory=list)
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    raw_model_output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingModelSession(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def inject_text(
        self,
        *,
        text: str,
        timestamp_sec: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def ingest_frame(
        self,
        *,
        frame: Any,
        timestamp_sec: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[StreamingOutputEvent]:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> list[StreamingOutputEvent]:
        raise NotImplementedError


class StreamingModelAdapter(ABC):
    @abstractmethod
    def create_session(
        self,
        *,
        input_video_path: str,
        frame_fps: float,
        adapter_settings: dict[str, Any],
        trace_metadata: dict[str, Any],
    ) -> StreamingModelSession:
        raise NotImplementedError

    def complete_qa(
        self,
        *,
        frames: list[Any],
        frame_timestamps: list[float],
        frame_fps: float,
        prompt: str,
        adapter_settings: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Forced one-shot QA decode: stream the given frames into the model's
        KV-cache, then generate a single answer to ``prompt`` (bypassing the
        proactive 'when to speak' gating). The model is loaded once and reused
        across calls. Returns the raw response text. Override per adapter."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support QA post-evaluation."
        )
