from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


NO_ACTION = "NO_ACTION"
VALID_TOOL_CALL = "VALID_TOOL_CALL"
VALID_OUTPUT = "VALID_OUTPUT"
INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
UNPARSEABLE_OUTPUT = "UNPARSEABLE_OUTPUT"


@dataclass
class NormalizedToolCall:
    name: str
    args: dict[str, Any]
    timestamp_sec: Optional[float] = None
    raw_entry: Any = None


@dataclass
class ParsedModelOutput:
    status: str
    raw_text: str
    parsed_payload: Any = None
    message: Optional[str] = None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    execution_allowed: bool = True
    issues: list[str] = field(default_factory=list)
    invalid_entries: list[Any] = field(default_factory=list)
    matched_silence_rule: Optional[str] = None


@dataclass
class InvalidActionRecord:
    reason: str
    tool_name: Optional[str]
    step_index: int
    timestamp_sec: Optional[float]
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransitionRecord:
    event_type: str
    tool_name: str
    interval_id: Optional[str]
    interval_type: Optional[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedInterval:
    interval_id: str
    interval_type: str
    tool_start_name: str
    tool_end_name: Optional[str]
    predicted_start_time: float
    predicted_end_time: Optional[float]
    is_open: bool
    instance_id: str
    args: dict[str, Any]
    start_step_index: int
    end_step_index: Optional[int]
    source_step_indices: list[int] = field(default_factory=list)
    validity_flags: list[str] = field(default_factory=list)


@dataclass
class StepRecord:
    step_index: int
    timestamp_sec: float
    sampled_frame_times: list[float]
    sampled_frame_fingerprints: list[str | None]
    sampled_frame_count: int
    request_messages: list[dict[str, Any]]
    raw_model_output: dict[str, Any]
    parser_status: str
    parser_issues: list[str]
    parser_invalid_entries: list[Any]
    parser_matched_silence_rule: Optional[str]
    parsed_payload: Any
    parsed_message: Optional[str]
    parsed_tool_calls: list[dict[str, Any]]
    requested_tool_calls: list[dict[str, Any]]
    executed_tool_calls: list[dict[str, Any]]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_transitions: list[StateTransitionRecord] = field(default_factory=list)
    invalid_actions: list[InvalidActionRecord] = field(default_factory=list)


@dataclass
class TickRecord:
    tick_index: int
    timestamp_sec: float
    sampled_frame_times: list[float]
    sampled_frame_fingerprints: list[str | None]
    output_event_indices: list[int] = field(default_factory=list)



@dataclass
class ExecutionTrace:
    trace_id: str
    source_unit_id: str
    evaluation_context: str
    input_mode: str
    input_video_path: str
    frame_sampling: dict[str, Any]
    output_events: list[StepRecord]
    predicted_intervals: list[PredictedInterval]
    metadata: dict[str, Any] = field(default_factory=dict)
    ticks: list[TickRecord] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    trace_schema_version: str = "proact.clip-trace.v1"

    @property
    def steps(self) -> list[StepRecord]:
        return self.output_events
