from __future__ import annotations

from typing import Any

from proact.evaluation.config import EvaluationConfig
from proact.evaluation.runtime.parser import ConfiguredResponseParser
from proact.evaluation.runtime.prompts import PromptBuilder
from proact.evaluation.runtime.records import ExecutionTrace, StepRecord, TickRecord
from proact.evaluation.runtime.state_tracker import ToolStateTracker
from proact.evaluation.runtime.streaming.base import StreamingModelAdapter, StreamingOutputEvent
from proact.evaluation.runtime.tools import ToolRegistry


class NativeStreamingExecutionBackend:
    def __init__(
        self,
        *,
        config: EvaluationConfig,
        parser: ConfiguredResponseParser,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        video_sampler: Any,
        adapter: StreamingModelAdapter,
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
        self.prompt_builder.reset_clip_cache()
        frames, frame_sampling_metadata = self.video_sampler.generate_stream_frames(
            input_video_path,
            fps=float(self.config.execution_backend.frame_fps or 0.0),
        )
        tracker = ToolStateTracker(
            self.tool_registry,
            message_history_config=self.config.prompt.message_history,
            prompt_context_config=self.config.prompt_context,
        )
        steps: list[StepRecord] = []
        ticks: list[TickRecord] = []
        next_step_index = 0
        injected_messages_log: list[dict[str, Any]] = []

        session = self.adapter.create_session(
            input_video_path=input_video_path,
            frame_fps=float(self.config.execution_backend.frame_fps or 0.0),
            adapter_settings=self.config.execution_backend.adapter_settings,
            trace_metadata=trace_metadata,
        )
        session.reset()

        if self.config.execution_backend.inject_prompt_as_initial_user_message:
            self._inject_session_text(
                session=session,
                injected_messages_log=injected_messages_log,
                text=self.prompt_builder.build_prompt_text(),
                timestamp_sec=0.0,
                metadata={"kind": "initial_prompt"},
            )

        if self._should_inject_tool_state():
            self._inject_session_text(
                session=session,
                injected_messages_log=injected_messages_log,
                text=self.prompt_builder.build_state_text(
                    tool_state=tracker.prompt_state(
                        state_format=self.config.prompt.state_format,
                        tool_state_templates=self.config.prompt.tool_state_templates,
                        include_tool_state=self.prompt_builder.include_tool_state_text(),
                    ),
                    timestamp_sec=0.0,
                ),
                timestamp_sec=0.0,
                metadata={"kind": "initial_tool_state"},
            )

        all_fingerprints = list(frame_sampling_metadata.get("sampled_frame_fingerprints") or [])
        for tick_index, frame in enumerate(frames):
            events = session.ingest_frame(
                frame=frame.array,
                timestamp_sec=float(frame.time),
                metadata={"chosen_for_time": getattr(frame, "chosen_for_time", float(frame.time))},
            )
            event_start_index = len(steps)
            next_step_index = self._consume_events(
                tracker=tracker,
                events=events,
                starting_step_index=next_step_index,
                steps=steps,
                session=session,
                injected_messages_log=injected_messages_log,
            )
            ticks.append(
                TickRecord(
                    tick_index=tick_index,
                    timestamp_sec=float(frame.time),
                    sampled_frame_times=[float(frame.time)],
                    sampled_frame_fingerprints=[
                        all_fingerprints[tick_index] if tick_index < len(all_fingerprints) else None
                    ],
                    output_event_indices=list(range(event_start_index, len(steps))),
                )
            )

        max_finalize_batches = max(1, len(frames) * 2)
        for _ in range(max_finalize_batches):
            final_events = session.finalize()
            if not final_events:
                break
            event_start_index = len(steps)
            next_step_index = self._consume_events(
                tracker=tracker,
                events=final_events,
                starting_step_index=next_step_index,
                steps=steps,
                session=session,
                injected_messages_log=injected_messages_log,
            )
            if ticks:
                ticks[-1].output_event_indices.extend(range(event_start_index, len(steps)))
        else:
            raise RuntimeError(
                f"Streaming adapter did not finish finalization within {max_finalize_batches} batches."
            )

        predicted_intervals = tracker.finalize()
        return ExecutionTrace(
            trace_id=trace_id,
            source_unit_id=source_unit_id,
            evaluation_context=self.config.context,
            input_mode="native-streaming",
            input_video_path=input_video_path,
            frame_sampling=frame_sampling_metadata
            | {
                "execution_backend": {
                    "mode": self.config.execution_backend.mode,
                    "adapter": self.config.execution_backend.adapter,
                    "frame_fps": self.config.execution_backend.frame_fps,
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
                }
            }
            | {
                "streaming_injected_messages": injected_messages_log,
            },
        )

    def _consume_events(
        self,
        *,
        tracker: ToolStateTracker,
        events: list[StreamingOutputEvent],
        starting_step_index: int,
        steps: list[StepRecord],
        session: Any,
        injected_messages_log: list[dict[str, Any]],
    ) -> int:
        next_step_index = starting_step_index
        for event in events:
            parsed_output = self.parser.parse(event.response_text)
            step = tracker.consume_step(
                parsed_output=parsed_output,
                step_index=next_step_index,
                step_time_sec=float(event.timestamp_sec),
                sampled_frame_times=list(event.sampled_frame_times or [float(event.timestamp_sec)]),
                request_messages=list(event.request_messages),
                raw_model_output=dict(event.raw_model_output) | {"response_text": event.response_text},
                allow_implicit_timestamps=True,
            )
            steps.append(step)
            next_step_index += 1

            if self._should_inject_tool_state() and step.state_before != step.state_after:
                self._inject_session_text(
                    session=session,
                    injected_messages_log=injected_messages_log,
                    text=self.prompt_builder.build_state_text(
                        tool_state=tracker.prompt_state(
                            state_format=self.config.prompt.state_format,
                            tool_state_templates=self.config.prompt.tool_state_templates,
                            include_tool_state=self.prompt_builder.include_tool_state_text(),
                        ),
                        timestamp_sec=float(event.timestamp_sec),
                    ),
                    timestamp_sec=float(event.timestamp_sec),
                    metadata={"kind": "tool_state_update", "source_step_index": next_step_index - 1},
                )
        return next_step_index

    def _inject_session_text(
        self,
        *,
        session: Any,
        injected_messages_log: list[dict[str, Any]],
        text: str,
        timestamp_sec: float,
        metadata: dict[str, Any],
    ) -> None:
        session.inject_text(
            text=text,
            timestamp_sec=float(timestamp_sec),
            metadata=dict(metadata),
        )
        injected_messages_log.append(
            {
                "role": "user",
                "content": text,
                "timestamp_sec": float(timestamp_sec),
                "metadata": dict(metadata),
            }
        )

    def _should_inject_tool_state(self) -> bool:
        return (
            self.config.execution_backend.inject_tool_state_updates
            and self.prompt_builder.include_state_text()
        )
