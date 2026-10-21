from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from proact.evaluation.config import PromptConfig
from proact.models.llm.base import ImageB64Part, LLMMessage, TextPart


_IMAGE_PART_CACHE_SIZE = 512


class PromptBuilder:
    def __init__(self, config: PromptConfig, prompt_context: Any | None = None):
        self.config = config
        self.prompt_context = prompt_context
        self.prompt_text = Path(config.prompt_path).read_text(encoding="utf-8")
        self._image_part_cache: OrderedDict[tuple[Any, ...], ImageB64Part] = OrderedDict()

    def build_state_text(
        self,
        *,
        tool_state: dict[str, Any],
        timestamp_sec: float,
    ) -> str:
        payload = {
            "current_time_sec": timestamp_sec,
            **tool_state,
        }
        return f"Current assistant context\n\n{json.dumps(payload, indent=2, sort_keys=True)}"

    def build_prompt_text(self) -> str:
        prompt_text = self.prompt_text.rstrip()
        response_format_prompt = self.config.response_format_prompt.strip()
        if response_format_prompt:
            prompt_text = f"{prompt_text}\n\n{response_format_prompt}"
        return prompt_text

    def include_state_text(self) -> bool:
        if self.prompt_context is not None:
            completed_interval_mode = getattr(
                getattr(self.prompt_context, "completed_tool_intervals", None),
                "mode",
                "none",
            )
            recent_tool_call_mode = getattr(
                getattr(self.prompt_context, "recent_tool_calls", None), "mode", "none"
            )
            recent_message_mode = getattr(
                getattr(self.prompt_context, "recent_messages", None), "mode", "none"
            )
            return bool(
                getattr(self.prompt_context, "include_active_tools", True)
                or completed_interval_mode != "none"
                or recent_tool_call_mode != "none"
                or recent_message_mode != "none"
            )
        return bool(self.config.include_state or self.config.message_history.enabled)

    def include_tool_state_text(self) -> bool:
        if self.prompt_context is not None:
            return bool(getattr(self.prompt_context, "include_active_tools", True))
        return bool(self.config.include_state)

    def reset_clip_cache(self) -> None:
        self._image_part_cache.clear()

    def build_messages(
        self,
        *,
        frames: list[Any],
        tool_state: dict[str, Any],
        step_index: int,
        timestamp_sec: float,
        evaluation_context: str,
        input_mode: str,
        input_video_path: str,
        metadata: dict[str, Any],
    ) -> list[LLMMessage]:
        prompt_text = self.build_prompt_text()

        content = [
            TextPart(text=prompt_text),
        ]
        if self.include_state_text():
            state_text = self.build_state_text(tool_state=tool_state, timestamp_sec=timestamp_sec)
            content.append(TextPart(text=state_text))
        content.append(TextPart(text="The window of video frames"))
        for index, frame in enumerate(frames):
            content.append(
                self._image_part_for_frame(
                    frame,
                    alt=None,
                )
            )
            content.append(TextPart(text=f"Frame #{index + 1} at {float(frame.time):.3f} seconds"))
        return [LLMMessage(role="user", content=content)]

    def _image_part_for_frame(self, frame: Any, *, alt: str | None) -> ImageB64Part:
        cache_key = self._image_cache_key(getattr(frame, "array", None))
        if cache_key is None:
            return ImageB64Part.from_source(frame.array, alt=alt)

        cached = self._image_part_cache.get(cache_key)
        if cached is not None:
            self._image_part_cache.move_to_end(cache_key)
            return ImageB64Part.from_source(cached, alt=alt)

        encoded = ImageB64Part.from_source(frame.array, alt=None)
        self._image_part_cache[cache_key] = encoded
        self._image_part_cache.move_to_end(cache_key)
        while len(self._image_part_cache) > _IMAGE_PART_CACHE_SIZE:
            self._image_part_cache.popitem(last=False)
        return ImageB64Part.from_source(encoded, alt=alt)

    @staticmethod
    def _image_cache_key(src: Any) -> tuple[Any, ...] | None:
        if isinstance(src, ImageB64Part):
            return ("image_b64", src.b64, src.mime_type or "")

        shape = getattr(src, "shape", None)
        dtype = getattr(src, "dtype", None)
        if shape is not None and dtype is not None and hasattr(src, "tobytes"):
            try:
                digest = hashlib.blake2b(src.tobytes(), digest_size=16).hexdigest()
            except Exception:
                return None
            return ("ndarray", tuple(shape), str(dtype), digest)

        return None
