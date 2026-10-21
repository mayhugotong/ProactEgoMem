from __future__ import annotations

import logging
import time
from typing import Any

from proact.evaluation.runtime.prompts import PromptBuilder
from proact.models.llm.base import LLMRequest

logger = logging.getLogger(__name__)


class LLMModelRunner:
    def __init__(self, llm, prompt_builder: PromptBuilder, request_params: dict[str, Any] | None = None):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.request_params = dict(request_params or {})
        self.json_mode = bool(self.request_params.pop("json_mode", False))

    def infer(
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
    ) -> dict[str, Any]:
        provider = getattr(getattr(self.llm, "config", None), "provider", None)
        model = getattr(getattr(self.llm, "config", None), "model", None)

        prepare_started_at = time.perf_counter()
        messages = self.prompt_builder.build_messages(
            frames=frames,
            tool_state=tool_state,
            step_index=step_index,
            timestamp_sec=timestamp_sec,
            evaluation_context=evaluation_context,
            input_mode=input_mode,
            input_video_path=input_video_path,
            metadata=metadata,
        )
        prepare_elapsed_s = time.perf_counter() - prepare_started_at
        logger.info(
            "LLM request prepared: provider=%s model=%s step_index=%s timestamp_sec=%.3f "
            "frames=%d elapsed_s=%.3f metadata=%s",
            provider,
            model,
            step_index,
            timestamp_sec,
            len(frames),
            prepare_elapsed_s,
            {
                key: metadata[key]
                for key in ("parent_video_uid", "match_id")
                if key in metadata
            },
        )
        request = LLMRequest(
            messages=messages,
            json_mode=self.json_mode,
            params=self.request_params,
            metadata={
                **metadata,
                "step_index": step_index,
                "timestamp_sec": timestamp_sec,
                "evaluation_context": evaluation_context,
                "input_mode": input_mode,
                "sampled_frame_count": len(frames),
            },
        )
        complete_started_at = time.perf_counter()
        response = self.llm.complete_one(request)
        complete_elapsed_s = time.perf_counter() - complete_started_at
        logger.info(
            "LLM request completed: provider=%s model=%s step_index=%s timestamp_sec=%.3f "
            "ok=%s elapsed_s=%.3f finish_reason=%s error_type=%s",
            provider,
            model,
            step_index,
            timestamp_sec,
            response.ok,
            complete_elapsed_s,
            response.finish_reason,
            response.error_type,
        )

        # The static instruction prompt repeats identically in every step; store
        # it once at trace level and record a marker here (~180x size saving).
        build_prompt_text = getattr(self.prompt_builder, "build_prompt_text", None)
        static_prompt = build_prompt_text() if callable(build_prompt_text) else ""

        def _recorded_content(message) -> str:
            content = message.to_string(include_role=False, include_alt=True)
            if static_prompt and content.startswith(static_prompt):
                return "[STATIC_PROMPT]" + content[len(static_prompt):]
            return content

        return {
            "request_messages": [
                {
                    "role": message.role,
                    "content": _recorded_content(message),
                }
                for message in messages
            ],
            "response_ok": response.ok,
            "response_text": response.text or "",
            "response_reasoning": response.reasoning,
            "response_finish_reason": response.finish_reason,
            "response_error": response.error,
            "response_error_type": response.error_type,
            "response_status_code": response.status_code,
            "response_raw": response.raw,
        }
