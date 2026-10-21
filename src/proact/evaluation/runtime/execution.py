from __future__ import annotations

import logging
import time
from typing import Any

from proact.evaluation.config import EvaluationConfig
from proact.evaluation.runtime.errors import ModelInferenceError
from proact.evaluation.runtime.model_runner import LLMModelRunner
from proact.evaluation.runtime.parser import ConfiguredResponseParser
from proact.evaluation.runtime.records import ExecutionTrace, TickRecord
from proact.evaluation.runtime.state_tracker import ToolStateTracker
from proact.evaluation.runtime.tools import ToolRegistry

logger = logging.getLogger(__name__)


class StatelessExecutionBackend:
    def __init__(
        self,
        *,
        config: EvaluationConfig,
        parser: ConfiguredResponseParser,
        tool_registry: ToolRegistry,
        model_runner: LLMModelRunner,
        video_sampler: Any,
    ) -> None:
        self.config = config
        self.parser = parser
        self.tool_registry = tool_registry
        self.model_runner = model_runner
        self.video_sampler = video_sampler

    def execute_trace(
        self,
        *,
        trace_id: str,
        source_unit_id: str,
        input_video_path: str,
        trace_metadata: dict[str, Any],
    ) -> ExecutionTrace:
        assert self.config.input_mode is not None
        prompt_builder = getattr(self.model_runner, "prompt_builder", None)
        if prompt_builder is not None:
            prompt_builder.reset_clip_cache()
        sampling_started_at = time.perf_counter()
        logger.info(
            "Frame sampling started: trace_id=%s video=%s mode=%s",
            trace_id,
            input_video_path,
            self.config.input_mode.mode,
        )
        batches, frame_sampling_metadata = self.video_sampler.generate_batches(
            input_video_path,
            self.config.input_mode,
        )
        sampling_elapsed_s = time.perf_counter() - sampling_started_at
        logger.info(
            "Frame sampling finished: trace_id=%s elapsed_s=%.3f backend=%s requested_backend=%s "
            "processing_path=%s gpu_resize=%s fallback_reason=%s batches=%d sampled_frames=%s resize=%s",
            trace_id,
            sampling_elapsed_s,
            (frame_sampling_metadata.get("backend") or {}).get("selected_backend"),
            (frame_sampling_metadata.get("backend") or {}).get("requested_backend"),
            (frame_sampling_metadata.get("backend") or {}).get("processing_path"),
            (frame_sampling_metadata.get("backend") or {}).get("gpu_resize"),
            (frame_sampling_metadata.get("backend") or {}).get("fallback_reason"),
            len(batches),
            frame_sampling_metadata.get("sampled_frame_count_total"),
            frame_sampling_metadata.get("frame_resize"),
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
                include_tool_state=(
                    prompt_builder.include_tool_state_text()
                    if prompt_builder is not None
                    else self.config.prompt.include_state
                ),
            )
            inference = self.model_runner.infer(
                frames=batch.frames,
                tool_state=prompt_state,
                step_index=batch.step_index,
                timestamp_sec=batch.timestamp_sec,
                evaluation_context=self.config.context,
                input_mode=self.config.input_mode.mode,
                input_video_path=input_video_path,
                metadata={
                    **trace_metadata,
                    "step_index": batch.step_index,
                    "batch_metadata": batch.metadata,
                },
            )
            if not inference.get("response_ok", True):
                # Provider/API errors must fail the clip (-> failures/<uid>.json),
                # never be parsed as model output and scored as silence.
                raise ModelInferenceError(
                    trace_id=trace_id,
                    step_index=batch.step_index,
                    timestamp_sec=batch.timestamp_sec,
                    error=inference.get("response_error"),
                    error_type=inference.get("response_error_type"),
                    status_code=inference.get("response_status_code"),
                    provider=str(trace_metadata.get("model") or "stateless"),
                )
            parsed_output = self.parser.parse(inference["response_text"] or "")
            step = tracker.consume_step(
                parsed_output=parsed_output,
                step_index=batch.step_index,
                step_time_sec=batch.timestamp_sec,
                sampled_frame_times=[float(frame.time) for frame in batch.frames],
                sampled_frame_fingerprints=list(batch.metadata.get("sampled_frame_fingerprints") or []),
                request_messages=inference["request_messages"],
                raw_model_output={
                    key: value
                    for key, value in inference.items()
                    if key not in {"request_messages", "response_text"}
                }
                | {"response_text": inference["response_text"]},
                allow_implicit_timestamps=self.config.input_mode.mode != "full-video",
            )
            steps.append(step)
            ticks.append(
                TickRecord(
                    tick_index=batch.step_index,
                    timestamp_sec=batch.timestamp_sec,
                    sampled_frame_times=[float(frame.time) for frame in batch.frames],
                    sampled_frame_fingerprints=list(
                        batch.metadata.get("sampled_frame_fingerprints") or []
                    ),
                    output_event_indices=[len(steps) - 1],
                )
            )

        predicted_intervals = tracker.finalize()
        static_prompt_text = (
            prompt_builder.build_prompt_text() if prompt_builder is not None else None
        )
        return ExecutionTrace(
            trace_id=trace_id,
            source_unit_id=source_unit_id,
            evaluation_context=self.config.context,
            input_mode=self.config.input_mode.mode,
            input_video_path=input_video_path,
            frame_sampling=frame_sampling_metadata
            | {
                "execution_backend": {
                    "mode": self.config.execution_backend.mode,
                    "adapter": self.config.execution_backend.adapter,
                }
            },
            output_events=steps,
            predicted_intervals=predicted_intervals,
            ticks=ticks,
            final_state=tracker.final_state(),
            metadata=trace_metadata
            | {
                "execution_backend": {
                    "mode": self.config.execution_backend.mode,
                    "adapter": self.config.execution_backend.adapter,
                },
                # Full text behind the per-step "[STATIC_PROMPT]" markers.
                "static_prompt_text": static_prompt_text,
            },
        )
