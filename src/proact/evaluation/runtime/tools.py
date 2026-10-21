from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    interval_type: str
    phase: str
    concurrency: str
    stop_arg_key: Optional[str] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise KeyError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def get(self, tool_name: str) -> ToolSpec:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool '{tool_name}'. Registered: {sorted(self._tools)}")
        return self._tools[tool_name]

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def names(self) -> set[str]:
        return set(self._tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="start_recording",
            interval_type="recording",
            phase="start",
            concurrency="single",
        )
    )
    registry.register(
        ToolSpec(
            name="stop_recording",
            interval_type="recording",
            phase="stop",
            concurrency="single",
        )
    )
    registry.register(
        ToolSpec(
            name="start_reminder",
            interval_type="reminder",
            phase="start",
            concurrency="multi",
        )
    )
    registry.register(
        ToolSpec(
            name="stop_reminder",
            interval_type="reminder",
            phase="stop",
            concurrency="multi",
            stop_arg_key="id",
        )
    )
    # Task-named reminder tools: identical semantics and interval_type as
    # start/stop_reminder, so traces carry interval_type='reminder' and
    # 'reminder-N' instance ids. start/stop_reminder stay registered so traces
    # that use them still replay.
    registry.register(
        ToolSpec(
            name="start_state_tracking",
            interval_type="reminder",
            phase="start",
            concurrency="multi",
        )
    )
    registry.register(
        ToolSpec(
            name="stop_state_tracking",
            interval_type="reminder",
            phase="stop",
            concurrency="multi",
            stop_arg_key="id",
        )
    )
    registry.register(
        ToolSpec(
            name="start_location_tracking",
            interval_type="reminder",
            phase="start",
            concurrency="multi",
        )
    )
    registry.register(
        ToolSpec(
            name="stop_location_tracking",
            interval_type="reminder",
            phase="stop",
            concurrency="multi",
            stop_arg_key="id",
        )
    )
    return registry
