from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from proact.evaluation.io.serialization import write_json, write_text


TRACE_FILENAME = "trace.json"
MEMORY_JSON_FILENAME = "proactive_memory.json"
MEMORY_MARKDOWN_FILENAME = "proactive_memory.md"
MANIFEST_FILENAME = "manifest.json"
MEMORY_SCHEMA_VERSION = "proact.proactive-memory.v1"
NO_ACTION_STATUS = "NO_ACTION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_proactive_memory(trace_payload: dict[str, Any]) -> dict[str, Any]:
    clip = trace_payload.get("clip")
    clip = clip if isinstance(clip, dict) else {}
    trace = trace_payload.get("trace")
    trace = trace if isinstance(trace, dict) else {}
    source_trace_schema = (
        trace_payload.get("trace_schema_version")
        or trace.get("trace_schema_version")
    )

    events = trace.get("output_events")
    events = events if isinstance(events, list) else []
    projected_events = [
        _project_output_event(event, event_index)
        for event_index, event in enumerate(events)
        if isinstance(event, dict)
    ]
    projected_events.sort(key=_event_sort_key)

    raw_model_outputs = [
        {
            "event_index": event["event_index"],
            "timestamp_sec": event["timestamp_sec"],
            "step_index": event["step_index"],
            "parser_status": event["parser_status"],
            "response_text": event["response_text"],
        }
        for event in projected_events
    ]
    all_output_events = [
        {key: value for key, value in event.items() if key != "response_text"}
        for event in projected_events
    ]
    non_silent_output_events = [
        event
        for event in all_output_events
        if event.get("parser_status") != NO_ACTION_STATUS
    ]

    intervals = trace.get("predicted_intervals")
    intervals = intervals if isinstance(intervals, list) else []
    collated_tool_intervals = [
        _project_interval(interval)
        for interval in intervals
        if isinstance(interval, dict)
    ]
    collated_tool_intervals.sort(
        key=lambda item: (
            _number(item.get("started_at_sec")),
            _integer(item.get("start_step_index")),
            str(item.get("instance_id") or item.get("interval_id") or ""),
        )
    )

    return {
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "source_trace_schema_version": source_trace_schema,
        "timestamp_semantics": {
            "unit": "seconds",
            "reference": "clip-local",
        },
        "clip": {
            "clip_uid": clip.get("clip_uid"),
            "video_uid": clip.get("video_uid"),
            "clip_duration_sec": clip.get("clip_duration_sec"),
        },
        "raw_model_outputs": raw_model_outputs,
        "all_output_events": all_output_events,
        "non_silent_output_events": non_silent_output_events,
        "collated_tool_intervals": collated_tool_intervals,
    }


def render_proactive_memory(memory: dict[str, Any]) -> str:
    clip = memory.get("clip")
    clip = clip if isinstance(clip, dict) else {}
    clip_uid = str(clip.get("clip_uid") or "unknown")
    lines = [
        f"# Proactive Memory: {clip_uid}",
        "",
        "All timestamps are clip-local seconds.",
        "",
        "## Collated Memory",
        "",
    ]

    intervals = memory.get("collated_tool_intervals")
    intervals = intervals if isinstance(intervals, list) else []
    if not intervals:
        lines.extend(["_No executed tool intervals._", ""])
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        start = _format_time(interval.get("started_at_sec"))
        end = (
            "OPEN"
            if interval.get("status") == "open" or interval.get("ended_at_sec") is None
            else _format_time(interval.get("ended_at_sec"))
        )
        lines.extend(
            [
                (
                    f"### {start} -> {end} | "
                    f"{interval.get('interval_type') or 'unknown'} | "
                    f"{interval.get('instance_id') or interval.get('interval_id') or 'unknown'}"
                ),
                "",
                f"- Start tool: `{interval.get('start_tool_name') or 'unknown'}`",
                f"- End tool: `{interval.get('end_tool_name') or 'OPEN'}`",
                f"- Start step: `{interval.get('start_step_index')}`",
                f"- End step: `{interval.get('end_step_index') if interval.get('end_step_index') is not None else 'OPEN'}`",
                "- Arguments:",
                _fenced_json(
                    interval.get("arguments")
                    if isinstance(interval.get("arguments"), dict)
                    else {}
                ),
                "",
            ]
        )

    events = memory.get("all_output_events")
    events = events if isinstance(events, list) else []
    event_dicts = [event for event in events if isinstance(event, dict)]

    lines.extend(["## User Messages", ""])
    messages = [
        event
        for event in event_dicts
        if isinstance(event.get("message"), str) and event["message"].strip()
    ]
    if not messages:
        lines.extend(["_No user-facing messages._", ""])
    for event in messages:
        lines.extend(
            [
                f"### {_event_heading(event)}",
                "",
                _fenced_text(event["message"]),
                "",
            ]
        )

    lines.extend(["## Raw Tool Calls", ""])
    non_silent = memory.get("non_silent_output_events")
    non_silent = non_silent if isinstance(non_silent, list) else []
    tool_events = [
        event
        for event in non_silent
        if isinstance(event, dict) and _has_projected_tool_call_data(event)
    ]
    if not tool_events:
        lines.extend(["_No tool-call data._", ""])
    for event in tool_events:
        lines.extend(
            [
                f"### {_event_heading(event)} | parser `{event.get('parser_status') or 'unknown'}`",
                "",
                "#### Original model tool-call entries",
                "",
                _fenced_json(event.get("original_tool_calls") or []),
                "",
                "#### Normalized requested calls",
                "",
                _fenced_json(event.get("requested_tool_calls") or []),
                "",
                "#### Executed calls",
                "",
                _fenced_json(event.get("executed_tool_calls") or []),
                "",
                "#### Parser-rejected entries",
                "",
                _fenced_json(event.get("parser_rejected_entries") or []),
                "",
                "#### Invalid actions",
                "",
                _fenced_json(event.get("invalid_actions") or []),
                "",
            ]
        )

    lines.extend(["## Raw Model Outputs", ""])
    raw_outputs = memory.get("raw_model_outputs")
    raw_outputs = raw_outputs if isinstance(raw_outputs, list) else []
    if not raw_outputs:
        lines.extend(["_No model-output events._", ""])
    for output in raw_outputs:
        if not isinstance(output, dict):
            continue
        response_text = output.get("response_text")
        if response_text is None:
            response_text = ""
        elif not isinstance(response_text, str):
            response_text = str(response_text)
        lines.extend(
            [
                f"### {_event_heading(output)} | parser `{output.get('parser_status') or 'unknown'}`",
                "",
                _fenced_text(response_text),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_clip_bundle(
    clip_dir: Path,
    *,
    trace_payload: dict[str, Any],
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
) -> dict[str, Any]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clip_dir / MANIFEST_FILENAME
    manifest_path.unlink(missing_ok=True)

    trace_path = clip_dir / TRACE_FILENAME
    memory_json_path = clip_dir / MEMORY_JSON_FILENAME
    memory_markdown_path = clip_dir / MEMORY_MARKDOWN_FILENAME
    write_json(trace_path, trace_payload)
    memory = build_proactive_memory(trace_payload)
    write_json(memory_json_path, memory)
    write_text(memory_markdown_path, render_proactive_memory(memory))
    manifest = _build_manifest(
        clip_dir,
        trace_payload=trace_payload,
        trace_schema_version=trace_schema_version,
        run_fingerprint=run_fingerprint,
        clip_catalog_fingerprint=clip_catalog_fingerprint,
    )
    write_json(manifest_path, manifest)
    return manifest


def inspect_clip_bundle(
    clip_dir: Path,
    *,
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
    clip_uid: str,
) -> tuple[str, dict[str, Any] | None]:
    trace_path = clip_dir / TRACE_FILENAME
    trace_payload = _load_valid_trace(
        trace_path,
        trace_schema_version=trace_schema_version,
        run_fingerprint=run_fingerprint,
        clip_catalog_fingerprint=clip_catalog_fingerprint,
        clip_uid=clip_uid,
    )
    if trace_payload is None:
        return "rerun", None

    manifest_path = clip_dir / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "repair-memory-json", trace_payload
    if not _manifest_identity_matches(
        manifest,
        trace_schema_version=trace_schema_version,
        run_fingerprint=run_fingerprint,
        clip_catalog_fingerprint=clip_catalog_fingerprint,
        clip_uid=clip_uid,
    ):
        return "repair-memory-json", trace_payload

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return "repair-memory-json", trace_payload
    trace_record = artifacts.get("trace")
    if (
        not isinstance(trace_record, dict)
        or trace_record.get("path") != TRACE_FILENAME
        or not isinstance(trace_record.get("sha256"), str)
    ):
        return "repair-memory-json", trace_payload
    if sha256_file(trace_path) != trace_record["sha256"]:
        return "rerun", None

    memory_json_path = clip_dir / MEMORY_JSON_FILENAME
    memory_json_record = artifacts.get("proactive_memory_json")
    memory = _load_valid_memory(
        memory_json_path,
        trace_payload=trace_payload,
        clip_uid=clip_uid,
    )
    if (
        memory is None
        or not _artifact_matches(
            memory_json_path,
            memory_json_record,
            MEMORY_JSON_FILENAME,
        )
    ):
        return "repair-memory-json", trace_payload

    memory_markdown_path = clip_dir / MEMORY_MARKDOWN_FILENAME
    memory_markdown_record = artifacts.get("proactive_memory_markdown")
    if not _artifact_matches(
        memory_markdown_path,
        memory_markdown_record,
        MEMORY_MARKDOWN_FILENAME,
    ):
        return "repair-markdown", trace_payload
    return "resume", trace_payload


def repair_clip_bundle(
    clip_dir: Path,
    *,
    trace_payload: dict[str, Any],
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
    rebuild_memory_json: bool,
) -> dict[str, Any]:
    manifest_path = clip_dir / MANIFEST_FILENAME
    manifest_path.unlink(missing_ok=True)
    clip = trace_payload.get("clip")
    clip = clip if isinstance(clip, dict) else {}
    memory_json_path = clip_dir / MEMORY_JSON_FILENAME
    memory = None if rebuild_memory_json else _load_valid_memory(
        memory_json_path,
        trace_payload=trace_payload,
        clip_uid=str(clip.get("clip_uid") or ""),
    )
    if memory is None:
        memory = build_proactive_memory(trace_payload)
        write_json(memory_json_path, memory)
    write_text(
        clip_dir / MEMORY_MARKDOWN_FILENAME,
        render_proactive_memory(memory),
    )
    manifest = _build_manifest(
        clip_dir,
        trace_payload=trace_payload,
        trace_schema_version=trace_schema_version,
        run_fingerprint=run_fingerprint,
        clip_catalog_fingerprint=clip_catalog_fingerprint,
    )
    write_json(manifest_path, manifest)
    return manifest


def _build_manifest(
    clip_dir: Path,
    *,
    trace_payload: dict[str, Any],
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
) -> dict[str, Any]:
    trace_path = clip_dir / TRACE_FILENAME
    memory_json_path = clip_dir / MEMORY_JSON_FILENAME
    memory_markdown_path = clip_dir / MEMORY_MARKDOWN_FILENAME
    clip = trace_payload.get("clip")
    clip = clip if isinstance(clip, dict) else {}
    return {
        "status": "complete",
        "trace_schema_version": trace_schema_version,
        "run_fingerprint": run_fingerprint,
        "clip_catalog_fingerprint": clip_catalog_fingerprint,
        "clip_uid": clip.get("clip_uid"),
        "artifacts": {
            "trace": {"path": TRACE_FILENAME, "sha256": sha256_file(trace_path)},
            "proactive_memory_json": {
                "path": MEMORY_JSON_FILENAME,
                "sha256": sha256_file(memory_json_path),
            },
            "proactive_memory_markdown": {
                "path": MEMORY_MARKDOWN_FILENAME,
                "sha256": sha256_file(memory_markdown_path),
            },
        },
    }


def _load_valid_trace(
    path: Path,
    *,
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
    clip_uid: str,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    trace = payload.get("trace")
    return payload if (
        payload.get("status") == "complete"
        and payload.get("trace_schema_version") == trace_schema_version
        and payload.get("run_fingerprint") == run_fingerprint
        and payload.get("clip_catalog_fingerprint") == clip_catalog_fingerprint
        and isinstance(payload.get("clip"), dict)
        and payload["clip"].get("clip_uid") == clip_uid
        and isinstance(trace, dict)
        and trace.get("trace_schema_version") == trace_schema_version
    ) else None


def _load_valid_memory(
    path: Path,
    *,
    trace_payload: dict[str, Any],
    clip_uid: str,
) -> dict[str, Any] | None:
    try:
        memory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    clip = memory.get("clip") if isinstance(memory, dict) else None
    if not isinstance(clip, dict):
        return None
    return memory if (
        memory.get("memory_schema_version") == MEMORY_SCHEMA_VERSION
        and memory.get("source_trace_schema_version")
        == trace_payload.get("trace_schema_version")
        and clip.get("clip_uid") == clip_uid
        and isinstance(memory.get("raw_model_outputs"), list)
        and isinstance(memory.get("all_output_events"), list)
        and isinstance(memory.get("non_silent_output_events"), list)
        and isinstance(memory.get("collated_tool_intervals"), list)
    ) else None


def _manifest_identity_matches(
    manifest: Any,
    *,
    trace_schema_version: str,
    run_fingerprint: str,
    clip_catalog_fingerprint: str,
    clip_uid: str,
) -> bool:
    return isinstance(manifest, dict) and (
        manifest.get("status") == "complete"
        and manifest.get("trace_schema_version") == trace_schema_version
        and manifest.get("run_fingerprint") == run_fingerprint
        and manifest.get("clip_catalog_fingerprint") == clip_catalog_fingerprint
        and manifest.get("clip_uid") == clip_uid
    )


def _artifact_matches(path: Path, record: Any, filename: str) -> bool:
    return (
        isinstance(record, dict)
        and record.get("path") == filename
        and isinstance(record.get("sha256"), str)
        and path.is_file()
        and sha256_file(path) == record["sha256"]
    )


def _project_output_event(event: dict[str, Any], event_index: int) -> dict[str, Any]:
    raw_output = event.get("raw_model_output")
    response_text = raw_output.get("response_text") if isinstance(raw_output, dict) else None
    if response_text is None:
        response_text = ""
    elif not isinstance(response_text, str):
        response_text = str(response_text)
    return {
        "event_index": event_index,
        "timestamp_sec": event.get("timestamp_sec"),
        "step_index": event.get("step_index"),
        "parser_status": event.get("parser_status"),
        "message": event.get("parsed_message"),
        "original_tool_calls": _raw_tool_call_entries(event.get("parsed_payload")),
        "parsed_tool_calls": event.get("parsed_tool_calls") or [],
        "requested_tool_calls": event.get("requested_tool_calls") or [],
        "executed_tool_calls": event.get("executed_tool_calls") or [],
        "parser_rejected_entries": event.get("parser_invalid_entries") or [],
        "invalid_actions": event.get("invalid_actions") or [],
        "parser_diagnostics": {
            "issues": event.get("parser_issues") or [],
            "matched_silence_rule": event.get("parser_matched_silence_rule"),
        },
        "response_text": response_text,
    }


def _project_interval(interval: dict[str, Any]) -> dict[str, Any]:
    ended_at_sec = interval.get("predicted_end_time")
    is_open = bool(interval.get("is_open")) or ended_at_sec is None
    return {
        "interval_id": interval.get("interval_id"),
        "instance_id": interval.get("instance_id"),
        "interval_type": interval.get("interval_type"),
        "start_tool_name": interval.get("tool_start_name"),
        "end_tool_name": interval.get("tool_end_name"),
        "arguments": interval.get("args") if isinstance(interval.get("args"), dict) else {},
        "started_at_sec": interval.get("predicted_start_time"),
        "ended_at_sec": None if is_open else ended_at_sec,
        "start_step_index": interval.get("start_step_index"),
        "end_step_index": None if is_open else interval.get("end_step_index"),
        "status": "open" if is_open else "closed",
    }


def _raw_tool_call_entries(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            return payload["tool_calls"]
        if "tools" in payload:
            return payload["tools"]
        if "tool" in payload:
            return payload["tool"]
    if isinstance(payload, list):
        return payload
    return []


def _has_projected_tool_call_data(event: dict[str, Any]) -> bool:
    return any(
        bool(event.get(key))
        for key in (
            "original_tool_calls",
            "parsed_tool_calls",
            "requested_tool_calls",
            "executed_tool_calls",
            "parser_rejected_entries",
            "invalid_actions",
        )
    )


def _event_sort_key(event: dict[str, Any]) -> tuple[float, int, int]:
    return (
        _number(event.get("timestamp_sec")),
        _integer(event.get("step_index")),
        _integer(event.get("event_index")),
    )


def _event_heading(event: dict[str, Any]) -> str:
    return f"{_format_time(event.get('timestamp_sec'))} | step {_integer(event.get('step_index'))}"


def _format_time(value: Any) -> str:
    number = _number(value)
    return "unknown" if number == float("inf") else f"{number:.3f}s"


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2**63 - 1


def _fenced_json(value: Any) -> str:
    return _fenced_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), "json")


def _fenced_text(text: str, language: str = "text") -> str:
    run = 0
    longest = 0
    for char in text:
        if char == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"
