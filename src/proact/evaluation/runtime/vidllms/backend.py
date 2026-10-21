from __future__ import annotations

import logging
from typing import Any

from proact.evaluation.config import EvaluationConfig
from proact.evaluation.runtime.errors import ModelInferenceError
from proact.evaluation.runtime.parser import ConfiguredResponseParser
from proact.evaluation.runtime.records import ExecutionTrace, TickRecord
from proact.evaluation.runtime.state_tracker import ToolStateTracker
from proact.evaluation.runtime.tools import ToolRegistry
from proact.evaluation.runtime.vidllms.base import (
    VideoLLMAdapter,
    VideoWindowRequest,
    build_interleaved_video_window_text,
)


logger = logging.getLogger(__name__)


class VideoLLMExecutionBackend:
    def __init__(
        self,
        *,
        config: EvaluationConfig,
        parser: ConfiguredResponseParser,
        tool_registry: ToolRegistry,
        prompt_builder: Any,
        video_sampler: Any,
        adapter: VideoLLMAdapter,
    ) -> None:
        self.config = config
        self.parser = parser
        self.tool_registry = tool_registry
        self.prompt_builder = prompt_builder
        self.video_sampler = video_sampler
        self.adapter = adapter

    def execute_trace(
        self,
        *,
        trace_id: str,
        source_unit_id: str,
        input_video_path: str,
        trace_metadata: dict[str, Any],
    ) -> ExecutionTrace:
        assert self.config.input_mode is not None
        self.prompt_builder.reset_clip_cache()
        batches, frame_sampling_metadata = self.video_sampler.generate_batches(
            input_video_path,
            self.config.input_mode,
        )
        tracker = ToolStateTracker(
            self.tool_registry,
            message_history_config=self.config.prompt.message_history,
            prompt_context_config=self.config.prompt_context,
        )
        steps = []
        ticks: list[TickRecord] = []

        for batch in batches:
            prompt_state = tracker.prompt_state(
                state_format=self.config.prompt.state_format,
                tool_state_templates=self.config.prompt.tool_state_templates,
                include_tool_state=self.prompt_builder.include_tool_state_text(),
            )
            prompt_text = self.prompt_builder.build_prompt_text()
            state_text = (
                self.prompt_builder.build_state_text(
                    tool_state=prompt_state,
                    timestamp_sec=batch.timestamp_sec,
                )
                if self.prompt_builder.include_state_text()
                else ""
            )
            frame_timestamps = [float(frame.time) for frame in batch.frames]
            input_text = self._request_content_for_trace(
                prompt_text=prompt_text,
                state_text=state_text,
                frame_timestamps=frame_timestamps,
            )
            # The adapter receives the full text; the recorded copy elides the
            # static prompt (stored once in trace metadata) behind a marker.
            recorded_text = (
                "[STATIC_PROMPT]" + input_text[len(prompt_text):]
                if prompt_text and input_text.startswith(prompt_text)
                else input_text
            )
            request_messages = [
                {
                    "role": "user",
                    "content": recorded_text,
                }
            ]
            # Adapter exceptions propagate: the clip must land in failures/ with a
            # report instead of being scored as model silence.
            response = self.adapter.complete_window(
                VideoWindowRequest(
                    input_video_path=input_video_path,
                    prompt_text=prompt_text,
                    state_text=state_text,
                    frames=batch.frames,
                    frame_timestamps=frame_timestamps,
                    step_index=batch.step_index,
                    timestamp_sec=batch.timestamp_sec,
                    input_mode=self.config.input_mode.mode,
                    input_text=input_text,
                    metadata={
                        **trace_metadata,
                        "step_index": batch.step_index,
                        "batch_metadata": batch.metadata,
                    },
                    adapter_settings=self.config.execution_backend.adapter_settings,
                )
            )
            if not response.response_ok:
                raise ModelInferenceError(
                    trace_id=trace_id,
                    step_index=batch.step_index,
                    timestamp_sec=batch.timestamp_sec,
                    error=response.response_error,
                    error_type=response.response_error_type,
                    status_code=response.response_status_code,
                    provider=f"vidllm:{self.config.execution_backend.adapter}",
                )
            parsed_output = self.parser.parse(response.response_text or "")
            raw_model_output = {
                "response_ok": response.response_ok,
                "response_text": response.response_text,
                "response_reasoning": response.response_reasoning,
                "response_finish_reason": response.response_finish_reason,
                "response_error": response.response_error,
                "response_error_type": response.response_error_type,
                "response_status_code": response.response_status_code,
                "response_raw": response.response_raw,
                **response.raw_model_output,
            }
            step = tracker.consume_step(
                parsed_output=parsed_output,
                step_index=batch.step_index,
                step_time_sec=batch.timestamp_sec,
                sampled_frame_times=frame_timestamps,
                sampled_frame_fingerprints=list(batch.metadata.get("sampled_frame_fingerprints") or []),
                request_messages=response.request_messages or request_messages,
                raw_model_output=raw_model_output,
                allow_implicit_timestamps=self.config.input_mode.mode != "full-video",
            )
            steps.append(step)
            ticks.append(
                TickRecord(
                    tick_index=batch.step_index,
                    timestamp_sec=batch.timestamp_sec,
                    sampled_frame_times=frame_timestamps,
                    sampled_frame_fingerprints=list(
                        batch.metadata.get("sampled_frame_fingerprints") or []
                    ),
                    output_event_indices=[len(steps) - 1],
                )
            )

        predicted_intervals = tracker.finalize()
        backend_metadata = {
            "mode": self.config.execution_backend.mode,
            "adapter": self.config.execution_backend.adapter,
        }
        return ExecutionTrace(
            trace_id=trace_id,
            source_unit_id=source_unit_id,
            evaluation_context=self.config.context,
            input_mode=self.config.input_mode.mode,
            input_video_path=input_video_path,
            frame_sampling=frame_sampling_metadata | {"execution_backend": backend_metadata},
            output_events=steps,
            predicted_intervals=predicted_intervals,
            ticks=ticks,
            final_state=tracker.final_state(),
            metadata=trace_metadata
            | {
                "execution_backend": backend_metadata,
                # Full text behind the per-step "[STATIC_PROMPT]" markers.
                "static_prompt_text": self.prompt_builder.build_prompt_text(),
            },
        )

    @staticmethod
    def _request_content_for_trace(
        *,
        prompt_text: str,
        state_text: str,
        frame_timestamps: list[float],
    ) -> str:
        return build_interleaved_video_window_text(
            prompt_text=prompt_text,
            state_text=state_text,
            frame_timestamps=frame_timestamps,
        )
