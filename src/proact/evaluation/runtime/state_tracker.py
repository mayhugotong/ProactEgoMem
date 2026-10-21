from __future__ import annotations

from collections import defaultdict
from string import Formatter
from typing import Any


from proact.evaluation.runtime.records import (
    NO_ACTION,
    InvalidActionRecord,
    ParsedModelOutput,
    PredictedInterval,
    StateTransitionRecord,
    StepRecord,
)
from proact.evaluation.runtime.tools import ToolRegistry


class ToolStateTracker:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        message_history_config: Any | None = None,
        prompt_context_config: Any | None = None,
    ):
        self.tool_registry = tool_registry
        self.message_history_config = message_history_config
        self.prompt_context_config = prompt_context_config
        self._instance_counters: defaultdict[str, int] = defaultdict(int)
        self._active_intervals: dict[str, PredictedInterval] = {}
        self._active_singletons: dict[str, str] = {}
        self._intervals: list[PredictedInterval] = []
        self._message_history: list[dict[str, Any]] = []
        self._attempted_tool_history: list[dict[str, Any]] = []
        self._executed_tool_history: list[dict[str, Any]] = []

    def snapshot_state(self) -> dict[str, Any]:
        active_intervals = sorted(
            self._active_intervals.values(),
            key=lambda interval: (interval.interval_type, interval.predicted_start_time, interval.instance_id),
        )
        return {
            "active_intervals": [
                {
                    "interval_id": interval.interval_id,
                    "instance_id": interval.instance_id,
                    "interval_type": interval.interval_type,
                    "tool_start_name": interval.tool_start_name,
                    "predicted_start_time": interval.predicted_start_time,
                    "args": dict(interval.args),
                    "source_step_indices": list(interval.source_step_indices),
                }
                for interval in active_intervals
            ]
        }

    def prompt_state(
        self,
        *,
        state_format: str,
        tool_state_templates: dict[str, Any] | None = None,
        include_tool_state: bool = True,
    ) -> dict[str, Any]:
        if state_format == "compact":
            return self._with_prompt_context(
                self._compact_prompt_state(include_active_tools=include_tool_state)
            )
        if not include_tool_state:
            return self._with_prompt_context({})
        if state_format == "full":
            return self._with_prompt_context(self.snapshot_state())
        if state_format == "active-only":
            return self._with_prompt_context(
                self._active_only_prompt_state(tool_state_templates or {})
            )
        raise ValueError(f"Unsupported prompt state format '{state_format}'.")

    def _compact_prompt_state(self, *, include_active_tools: bool) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if include_active_tools:
            active_intervals = sorted(
                self._active_intervals.values(),
                key=lambda interval: (
                    interval.interval_type,
                    interval.predicted_start_time,
                    interval.instance_id,
                ),
            )
            state["active_tools"] = [
                self._serialize_active_interval(interval) for interval in active_intervals
            ]

        selection = getattr(self.prompt_context_config, "completed_tool_intervals", None)
        mode = getattr(selection, "mode", "none") if selection is not None else "none"
        if mode != "none":
            completed = sorted(
                (interval for interval in self._intervals if not interval.is_open),
                key=lambda interval: (
                    float(interval.predicted_end_time or interval.predicted_start_time),
                    interval.instance_id,
                ),
            )
            if mode == "last_n":
                completed = completed[-int(getattr(selection, "n", 5)) :]
            state["completed_tool_intervals"] = [
                self._serialize_completed_interval(interval) for interval in completed
            ]
        return state

    def _active_only_prompt_state(self, tool_state_templates: dict[str, Any]) -> dict[str, Any]:
        messages: list[str] = []
        active_intervals = list(self._active_intervals.values())

        for tool_name, template in tool_state_templates.items():
            spec = self.tool_registry.get(tool_name)
            if spec.phase == "start":
                matching_intervals = [
                    interval for interval in active_intervals if interval.tool_start_name == tool_name
                ]
                for interval in matching_intervals:
                    messages.append(
                        self._render_active_only_message(
                            tool_name=tool_name,
                            template=template,
                            args=interval.args,
                            instance_id=interval.instance_id,
                            start_time_sec=interval.predicted_start_time,
                        )
                    )
            else:
                has_active_for_interval_type = any(
                    interval.interval_type == spec.interval_type for interval in active_intervals
                )
                if not has_active_for_interval_type:
                    messages.append(
                        self._render_active_only_message(
                            tool_name=tool_name,
                            template=template,
                            args={},
                            instance_id=None,
                            start_time_sec=None,
                        )
                    )

        return {"state_messages": messages}

    def consume_step(
        self,
        *,
        parsed_output: ParsedModelOutput,
        step_index: int,
        step_time_sec: float,
        sampled_frame_times: list[float],
        request_messages: list[dict[str, Any]],
        raw_model_output: dict[str, Any],
        allow_implicit_timestamps: bool,
        sampled_frame_fingerprints: list[str | None] | None = None,
    ) -> StepRecord:
        state_before = self._with_prompt_context(self.snapshot_state())
        transitions: list[StateTransitionRecord] = []
        invalid_actions: list[InvalidActionRecord] = []

        requested_tool_calls = [
            {
                "name": tool_call.name,
                "arguments": dict(tool_call.args),
                "timestamp_sec": tool_call.timestamp_sec,
            }
            for tool_call in parsed_output.tool_calls
        ]
        for item in requested_tool_calls:
            self._attempted_tool_history.append(
                item | {"step_index": step_index, "event_timestamp_sec": float(step_time_sec)}
            )

        executed_tool_calls: list[dict[str, Any]] = []
        calls_to_execute = parsed_output.tool_calls if parsed_output.execution_allowed else []
        for tool_call in calls_to_execute:
            effective_timestamp = (
                tool_call.timestamp_sec
                if tool_call.timestamp_sec is not None
                else (step_time_sec if allow_implicit_timestamps else None)
            )
            if effective_timestamp is None:
                invalid_actions.append(
                    InvalidActionRecord(
                        reason="Tool call is missing an explicit timestamp in a mode that requires one.",
                        tool_name=tool_call.name,
                        step_index=step_index,
                        timestamp_sec=None,
                        args=dict(tool_call.args),
                    )
                )
                continue

            if not self.tool_registry.has(tool_call.name):
                invalid_actions.append(
                    InvalidActionRecord(
                        reason="Tool name is not registered.",
                        tool_name=tool_call.name,
                        step_index=step_index,
                        timestamp_sec=effective_timestamp,
                        args=dict(tool_call.args),
                    )
                )
                continue

            spec = self.tool_registry.get(tool_call.name)
            transition_count_before = len(transitions)
            if spec.phase == "start":
                self._handle_start(
                    tool_name=tool_call.name,
                    interval_type=spec.interval_type,
                    concurrency=spec.concurrency,
                    timestamp_sec=effective_timestamp,
                    args=tool_call.args,
                    step_index=step_index,
                    transitions=transitions,
                    invalid_actions=invalid_actions,
                )
            else:
                self._handle_stop(
                    tool_name=tool_call.name,
                    interval_type=spec.interval_type,
                    concurrency=spec.concurrency,
                    stop_arg_key=spec.stop_arg_key,
                    timestamp_sec=effective_timestamp,
                    args=tool_call.args,
                    step_index=step_index,
                    transitions=transitions,
                    invalid_actions=invalid_actions,
                )
            if len(transitions) > transition_count_before:
                executed = {
                    "name": tool_call.name,
                    "arguments": dict(tool_call.args),
                    "timestamp_sec": float(effective_timestamp),
                    "step_index": step_index,
                }
                executed_tool_calls.append(executed)
                self._executed_tool_history.append(dict(executed))

        self._record_message_history(
            parsed_output=parsed_output,
            step_index=step_index,
            timestamp_sec=step_time_sec,
        )
        state_after = self._with_prompt_context(self.snapshot_state())

        return StepRecord(
            step_index=step_index,
            timestamp_sec=step_time_sec,
            sampled_frame_times=sampled_frame_times,
            sampled_frame_fingerprints=list(sampled_frame_fingerprints or []),
            sampled_frame_count=len(sampled_frame_times),
            request_messages=request_messages,
            raw_model_output=raw_model_output,
            parser_status=parsed_output.status,
            parser_issues=list(parsed_output.issues),
            parser_invalid_entries=list(parsed_output.invalid_entries),
            parser_matched_silence_rule=parsed_output.matched_silence_rule,
            parsed_payload=parsed_output.parsed_payload,
            parsed_message=parsed_output.message,
            parsed_tool_calls=[
                {
                    "name": tool_call.name,
                    "args": dict(tool_call.args),
                    "timestamp_sec": tool_call.timestamp_sec,
                }
                for tool_call in parsed_output.tool_calls
            ],
            requested_tool_calls=requested_tool_calls,
            executed_tool_calls=executed_tool_calls,
            state_before=state_before,
            state_after=state_after,
            state_transitions=transitions,
            invalid_actions=invalid_actions,
        )

    def _message_history_enabled(self) -> bool:
        selection = getattr(self.prompt_context_config, "recent_messages", None)
        if selection is not None:
            return getattr(selection, "mode", "none") != "none"
        return bool(getattr(self.message_history_config, "enabled", False))

    def _message_history_max_messages(self) -> int:
        selection = getattr(self.prompt_context_config, "recent_messages", None)
        if selection is not None and getattr(selection, "mode", "none") == "last_n":
            return int(getattr(selection, "n", 10))
        if selection is not None and getattr(selection, "mode", "none") == "all":
            return max(1, len(self._message_history) + 1)
        return int(getattr(self.message_history_config, "max_messages", 10))

    def _message_history_message_key(self) -> str:
        return str(getattr(self.message_history_config, "message_key", "message"))

    def _message_history_include_silence(self) -> bool:
        return bool(getattr(self.message_history_config, "include_silence", False))

    def _with_prompt_context(self, state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state)
        if self._message_history_enabled():
            result["recent_messages"] = [dict(item) for item in self._message_history]

        selection = getattr(self.prompt_context_config, "recent_tool_calls", None)
        mode = getattr(selection, "mode", "none") if selection is not None else "none"
        if mode != "none":
            source = getattr(selection, "source", "executed")
            history = (
                self._attempted_tool_history
                if source == "attempted"
                else self._executed_tool_history
            )
            if mode == "last_n":
                history = history[-int(getattr(selection, "n", 10)) :]
            result["recent_tool_calls"] = [
                self._serialize_recent_tool_call(item) for item in history
            ]
        return result

    def _record_message_history(
        self,
        *,
        parsed_output: ParsedModelOutput,
        step_index: int,
        timestamp_sec: float,
    ) -> None:
        if not self._message_history_enabled():
            return
        if parsed_output.status == NO_ACTION and not self._message_history_include_silence():
            return

        message = parsed_output.message
        if message is None:
            message = self._lookup_message_history_value(
                parsed_output.parsed_payload,
                self._message_history_message_key(),
            )
        if not isinstance(message, str) or not message.strip():
            return

        self._message_history.append(
            {
                "timestamp_sec": float(timestamp_sec),
                "step_index": int(step_index),
                "message": message,
            }
        )
        max_messages = self._message_history_max_messages()
        if len(self._message_history) > max_messages:
            self._message_history = self._message_history[-max_messages:]

    @staticmethod
    def _lookup_message_history_value(payload: Any, path: str) -> Any:
        current = payload
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            elif isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            else:
                return None
        return current

    def final_state(self) -> dict[str, Any]:
        return self.snapshot_state() | {
            "attempted_tool_calls": [dict(item) for item in self._attempted_tool_history],
            "executed_tool_calls": [dict(item) for item in self._executed_tool_history],
            "messages": [dict(item) for item in self._message_history],
        }

    def finalize(self) -> list[PredictedInterval]:
        return list(self._intervals)

    def _handle_start(
        self,
        *,
        tool_name: str,
        interval_type: str,
        concurrency: str,
        timestamp_sec: float,
        args: dict[str, Any],
        step_index: int,
        transitions: list[StateTransitionRecord],
        invalid_actions: list[InvalidActionRecord],
    ) -> None:
        if concurrency == "single" and interval_type in self._active_singletons:
            invalid_actions.append(
                InvalidActionRecord(
                    reason=f"Cannot call {tool_name} while a {interval_type} interval is already active.",
                    tool_name=tool_name,
                    step_index=step_index,
                    timestamp_sec=timestamp_sec,
                    args=dict(args),
                )
            )
            return

        instance_id = self._next_instance_id(interval_type)
        interval = PredictedInterval(
            interval_id=instance_id,
            interval_type=interval_type,
            tool_start_name=tool_name,
            tool_end_name=None,
            predicted_start_time=timestamp_sec,
            predicted_end_time=None,
            is_open=True,
            instance_id=instance_id,
            args=dict(args),
            start_step_index=step_index,
            end_step_index=None,
            source_step_indices=[step_index],
        )
        self._active_intervals[instance_id] = interval
        self._intervals.append(interval)
        if concurrency == "single":
            self._active_singletons[interval_type] = instance_id

        transitions.append(
            StateTransitionRecord(
                event_type="open_interval",
                tool_name=tool_name,
                interval_id=interval.interval_id,
                interval_type=interval.interval_type,
                details={"instance_id": instance_id, "args": dict(args)},
            )
        )

    def _handle_stop(
        self,
        *,
        tool_name: str,
        interval_type: str,
        concurrency: str,
        stop_arg_key: str | None,
        timestamp_sec: float,
        args: dict[str, Any],
        step_index: int,
        transitions: list[StateTransitionRecord],
        invalid_actions: list[InvalidActionRecord],
    ) -> None:
        if concurrency == "single":
            instance_id = self._active_singletons.get(interval_type)
            if instance_id is None:
                invalid_actions.append(
                    InvalidActionRecord(
                        reason=f"Cannot call {tool_name} because no {interval_type} interval is active.",
                        tool_name=tool_name,
                        step_index=step_index,
                        timestamp_sec=timestamp_sec,
                        args=dict(args),
                    )
                )
                return
        else:
            if stop_arg_key is None:
                raise RuntimeError(f"Tool {tool_name} requires a stop_arg_key.")
            target_id = args.get(stop_arg_key)
            if not isinstance(target_id, str) or not target_id:
                invalid_actions.append(
                    InvalidActionRecord(
                        reason=f"Tool {tool_name} requires a non-empty '{stop_arg_key}' argument.",
                        tool_name=tool_name,
                        step_index=step_index,
                        timestamp_sec=timestamp_sec,
                        args=dict(args),
                    )
                )
                return
            instance_id = target_id
            if instance_id not in self._active_intervals:
                invalid_actions.append(
                    InvalidActionRecord(
                        reason=f"Cannot call {tool_name} for unknown or inactive id '{instance_id}'.",
                        tool_name=tool_name,
                        step_index=step_index,
                        timestamp_sec=timestamp_sec,
                        args=dict(args),
                    )
                )
                return

        interval = self._active_intervals.pop(instance_id)
        interval.predicted_end_time = timestamp_sec
        interval.tool_end_name = tool_name
        interval.end_step_index = step_index
        interval.is_open = False
        interval.source_step_indices.append(step_index)
        if concurrency == "single":
            self._active_singletons.pop(interval_type, None)

        transitions.append(
            StateTransitionRecord(
                event_type="close_interval",
                tool_name=tool_name,
                interval_id=interval.interval_id,
                interval_type=interval.interval_type,
                details={"instance_id": interval.instance_id, "args": dict(args)},
            )
        )

    def _next_instance_id(self, interval_type: str) -> str:
        self._instance_counters[interval_type] += 1
        return f"{interval_type}-{self._instance_counters[interval_type]}"

    @staticmethod
    def _serialize_active_interval(interval: PredictedInterval) -> dict[str, Any]:
        return {
            "instance_id": interval.instance_id,
            "tool_name": interval.tool_start_name,
            "started_at_sec": interval.predicted_start_time,
            "arguments": dict(interval.args),
        }

    @staticmethod
    def _serialize_completed_interval(interval: PredictedInterval) -> dict[str, Any]:
        return {
            "instance_id": interval.instance_id,
            "start_tool_name": interval.tool_start_name,
            "end_tool_name": interval.tool_end_name,
            "started_at_sec": interval.predicted_start_time,
            "ended_at_sec": interval.predicted_end_time,
            "arguments": dict(interval.args),
        }

    @staticmethod
    def _serialize_recent_tool_call(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool_name": item["name"],
            "arguments": dict(item.get("arguments", {})),
            "called_at_sec": item.get("timestamp_sec"),
        }

    def _serialize_interval_for_prompt(self, interval: PredictedInterval) -> dict[str, Any]:
        result = {
            "instance_id": interval.instance_id,
            "interval_type": interval.interval_type,
            "tool_name": interval.tool_start_name,
            "args": dict(interval.args),
            "start_timestamp_sec": interval.predicted_start_time,
            "status": "open" if interval.is_open else "closed",
        }
        if interval.predicted_end_time is not None:
            result["end_timestamp_sec"] = interval.predicted_end_time
        return result

    def _render_active_only_message(
        self,
        *,
        tool_name: str,
        template: Any,
        args: dict[str, Any],
        instance_id: str | None,
        start_time_sec: float | None,
    ) -> str:
        template_message = getattr(template, "message", None)
        forwarded_args = getattr(template, "forwarded_args", None)
        if template_message is None and isinstance(template, dict):
            template_message = template.get("message", "")
        if forwarded_args is None and isinstance(template, dict):
            forwarded_args = template.get("forwarded_args", [])
        forwarded_args = list(forwarded_args or [])

        values: dict[str, Any] = {
            "id": instance_id or "",
            "instance_id": instance_id or "",
            "start_time_sec": start_time_sec if start_time_sec is not None else "",
            "start_timestamp_sec": start_time_sec if start_time_sec is not None else "",
        }
        for _, field_name, _, _ in Formatter().parse(template_message):
            if not field_name:
                continue
            if field_name in values:
                continue
            if field_name not in forwarded_args:
                continue
            if field_name not in args:
                print(
                    f"[prompt-state] Missing forwarded arg '{field_name}' for tool '{tool_name}'"
                    + (f" on instance '{instance_id}'." if instance_id else "."),
                    flush=True,
                )
                values[field_name] = ""
                continue
            values[field_name] = args[field_name]

        return template_message.format(**values)
