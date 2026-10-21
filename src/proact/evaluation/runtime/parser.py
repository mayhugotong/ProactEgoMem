from __future__ import annotations

import json
import re
from typing import Any, Iterable

from proact.evaluation.config import ParserConfig
from proact.evaluation.runtime.records import (
    INVALID_TOOL_CALL,
    NO_ACTION,
    UNPARSEABLE_OUTPUT,
    VALID_TOOL_CALL,
    VALID_OUTPUT,
    NormalizedToolCall,
    ParsedModelOutput,
)


_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _lookup_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                raise KeyError(path)
            index = int(part)
            current = current[index]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(path)
    return current


def _iter_json_values(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Any]] = []
    for match in re.finditer(r"[\{\[]", text):
        try:
            payload, offset = decoder.raw_decode(text[match.start():])
            candidates.append((match.start(), match.start() + offset, payload))
        except json.JSONDecodeError:
            continue
    return candidates


def _extract_first_json_value(text: str) -> Any:
    candidates = _iter_json_values(text)
    if candidates:
        return candidates[0][2]
    raise json.JSONDecodeError("No JSON object or array found", text, 0)


def _extract_last_json_value(text: str) -> Any:
    candidates = _iter_json_values(text)
    if not candidates:
        raise json.JSONDecodeError("No JSON object or array found", text, 0)
    _, _, payload = max(candidates, key=lambda candidate: (candidate[1], -candidate[0]))
    return payload


def _load_json_payload(
    text: str,
    *,
    allow_code_fence_json: bool,
    allow_first_json_match: bool = False,
    allow_last_json_match: bool = False,
) -> Any:
    payload_text = text
    if allow_code_fence_json:
        match = _CODE_BLOCK_RE.match(text)
        if match:
            payload_text = match.group(1).strip()

    try:
        return json.loads(payload_text)
    except json.JSONDecodeError as exc:
        try:
            if allow_first_json_match:
                return _extract_first_json_value(text)
            if allow_last_json_match:
                return _extract_last_json_value(text)
        except json.JSONDecodeError:
            pass
        raise exc


class ConfiguredResponseParser:
    def __init__(
        self,
        config: ParserConfig,
        allowed_tool_names: Iterable[str],
        proactive_level: str = "intervention",
    ):
        self.config = config
        self.allowed_tool_names = set(allowed_tool_names)
        self.proactive_level = proactive_level

    def parse(self, raw_text: str) -> ParsedModelOutput:
        if self.config.unified_schema:
            return self._parse_unified(raw_text)
        return self._parse_legacy(raw_text)

    def _parse_legacy(self, raw_text: str) -> ParsedModelOutput:
        text = (raw_text or "").strip()
        if text in self.config.silence.raw_text_values:
            return ParsedModelOutput(
                status=NO_ACTION,
                raw_text=raw_text,
                matched_silence_rule="raw_text",
            )

        try:
            payload = _load_json_payload(
                text,
                allow_code_fence_json=self.config.allow_code_fence_json,
                allow_first_json_match=self.config.allow_first_json_match,
                allow_last_json_match=self.config.allow_last_json_match,
            )
        except json.JSONDecodeError as exc:
            return ParsedModelOutput(
                status=UNPARSEABLE_OUTPUT,
                raw_text=raw_text,
                issues=[f"JSON decode error: {exc.msg}"],
            )

        for rule in self.config.silence.json_value_rules:
            try:
                value = _lookup_path(payload, rule.path)
            except (KeyError, IndexError, TypeError):
                continue
            if value == rule.equals:
                return ParsedModelOutput(
                    status=NO_ACTION,
                    raw_text=raw_text,
                    parsed_payload=payload,
                    matched_silence_rule=rule.path,
                )

        if (
            self.config.silence.empty_list_means_no_action
            and isinstance(payload, list)
            and not payload
        ):
            return ParsedModelOutput(
                status=NO_ACTION,
                raw_text=raw_text,
                parsed_payload=payload,
                matched_silence_rule="empty_list",
            )

        entries = self._extract_entries(payload)
        if not entries:
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=["Parsed JSON but found no configured tool-call payload."],
            )

        tool_calls: list[NormalizedToolCall] = []
        issues: list[str] = []
        invalid_entries: list[Any] = []

        for entry in entries:
            call, entry_issues = self._normalize_entry(entry)
            if call is not None:
                tool_calls.append(call)
            if entry_issues:
                issues.extend(entry_issues)
                invalid_entries.append(entry)

        if not tool_calls and issues:
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=issues,
                invalid_entries=invalid_entries,
            )

        status = VALID_TOOL_CALL if not issues else INVALID_TOOL_CALL
        return ParsedModelOutput(
            status=status,
            raw_text=raw_text,
            parsed_payload=payload,
            tool_calls=tool_calls,
            issues=issues,
            invalid_entries=invalid_entries,
        )

    def _parse_unified(self, raw_text: str) -> ParsedModelOutput:
        if self.proactive_level == "notification":
            return self._parse_notification_unified(raw_text)

        text = (raw_text or "").strip()
        try:
            payload = _load_json_payload(
                text,
                allow_code_fence_json=self.config.allow_code_fence_json,
                allow_first_json_match=self.config.json_match == "first",
                allow_last_json_match=self.config.json_match == "last",
            )
        except json.JSONDecodeError as exc:
            return ParsedModelOutput(
                status=UNPARSEABLE_OUTPUT,
                raw_text=raw_text,
                issues=[f"JSON decode error: {exc.msg}"],
                execution_allowed=False,
            )

        if not isinstance(payload, dict) or set(payload) != {"message", "tool_calls"}:
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=["Expected exactly the root keys 'message' and 'tool_calls'."],
                execution_allowed=False,
            )

        message = payload["message"]
        if message is not None and not isinstance(message, str):
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=["'message' must be a string or null."],
                execution_allowed=False,
            )
        entries = payload["tool_calls"]
        if not isinstance(entries, list):
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                message=message,
                issues=["'tool_calls' must be a list."],
                execution_allowed=False,
            )

        tool_calls: list[NormalizedToolCall] = []
        issues: list[str] = []
        invalid_entries: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"name", "arguments"}:
                issues.append("Each tool call must contain exactly 'name' and 'arguments'.")
                invalid_entries.append(entry)
                continue
            name = entry.get("name")
            arguments = entry.get("arguments")
            if not isinstance(name, str) or not name:
                issues.append("Tool call name must be a non-empty string.")
                invalid_entries.append(entry)
                continue
            if not isinstance(arguments, dict):
                issues.append(f"Tool '{name}' arguments must be an object.")
                invalid_entries.append(entry)
                continue
            if name not in self.allowed_tool_names:
                issues.append(f"Tool '{name}' is not allowed by the active task/tool config.")
                invalid_entries.append(entry)
                continue
            tool_calls.append(
                NormalizedToolCall(
                    name=name,
                    args=arguments,
                    raw_entry=entry,
                )
            )

        has_message = isinstance(message, str) and bool(message.strip())
        if not issues and not has_message and not tool_calls:
            status = NO_ACTION
        elif issues:
            status = INVALID_TOOL_CALL
        else:
            status = VALID_OUTPUT
        # Validity is per tool call: malformed/disallowed entries are recorded in
        # `issues`/`invalid_entries` and skipped, while their valid siblings still
        # execute. A step is never voided wholesale by one bad entry.
        return ParsedModelOutput(
            status=status,
            raw_text=raw_text,
            parsed_payload=payload,
            message=message,
            tool_calls=tool_calls,
            issues=issues,
            invalid_entries=invalid_entries,
            execution_allowed=True,
            matched_silence_rule="unified_empty" if status == NO_ACTION else None,
        )

    def _parse_notification_unified(self, raw_text: str) -> ParsedModelOutput:
        text = (raw_text or "").strip()
        if not text and self.config.allow_notification_raw_text:
            return ParsedModelOutput(
                status=NO_ACTION,
                raw_text=raw_text,
                parsed_payload={"message": None},
                matched_silence_rule="notification_empty_text",
            )

        try:
            payload = _load_json_payload(
                text,
                allow_code_fence_json=self.config.allow_code_fence_json,
                allow_first_json_match=self.config.json_match == "first",
                allow_last_json_match=self.config.json_match == "last",
            )
        except json.JSONDecodeError as exc:
            if (
                self.config.allow_notification_raw_text
                and text
                and not text.startswith(("{", "[", '"', "```"))
            ):
                return ParsedModelOutput(
                    status=VALID_OUTPUT,
                    raw_text=raw_text,
                    parsed_payload={"message": text},
                    message=text,
                )
            return ParsedModelOutput(
                status=UNPARSEABLE_OUTPUT,
                raw_text=raw_text,
                issues=[f"JSON decode error: {exc.msg}"],
                execution_allowed=False,
            )

        if not isinstance(payload, dict) or set(payload) != {"message"}:
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=["Expected exactly the root key 'message'."],
                execution_allowed=False,
            )

        message = payload["message"]
        if message is not None and not isinstance(message, str):
            return ParsedModelOutput(
                status=INVALID_TOOL_CALL,
                raw_text=raw_text,
                parsed_payload=payload,
                issues=["'message' must be a string or null."],
                execution_allowed=False,
            )

        has_message = isinstance(message, str) and bool(message.strip())
        return ParsedModelOutput(
            status=VALID_OUTPUT if has_message else NO_ACTION,
            raw_text=raw_text,
            parsed_payload=payload,
            message=message,
            matched_silence_rule=None if has_message else "notification_empty_message",
        )

    def _extract_entries(self, payload: Any) -> list[Any]:
        if self.config.allow_root_list and isinstance(payload, list):
            return list(payload)

        if not isinstance(payload, dict):
            return []

        for key in self.config.multi_tool_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)

        for key in self.config.single_tool_keys:
            value = payload.get(key)
            if value is not None:
                return [value]

        if self.config.allow_root_tool_object:
            return [payload]

        return []

    def _normalize_entry(self, entry: Any) -> tuple[NormalizedToolCall | None, list[str]]:
        issues: list[str] = []
        if not isinstance(entry, dict):
            return None, [f"Tool entry must be a dict, got {type(entry).__name__}."]

        tool_payload = entry
        for key in self.config.entry_tool_keys:
            nested = entry.get(key)
            if nested is not None:
                tool_payload = nested
                break

        if not isinstance(tool_payload, dict):
            return None, ["Configured nested tool payload is not a dict."]

        tool_name = tool_payload.get(self.config.tool_name_key)
        if not isinstance(tool_name, str) or not tool_name:
            issues.append("Tool call is missing a non-empty tool name.")
            return None, issues

        normalized_tool_name = tool_name
        if normalized_tool_name not in self.allowed_tool_names:
            issues.append(f"Tool '{tool_name}' is not allowed by the active task/tool config.")

        args = tool_payload.get(self.config.tool_args_key, {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            issues.append(f"Tool '{tool_name}' args must be a dict.")
            return None, issues
        normalized_args = args

        timestamp_sec = None
        for key in self.config.timestamp_keys:
            source = tool_payload if key in tool_payload else entry
            if key not in source:
                continue
            value = source[key]
            try:
                timestamp_sec = float(value)
            except (TypeError, ValueError):
                issues.append(f"Tool '{tool_name}' timestamp at key '{key}' is not numeric.")
            break

        return (
            NormalizedToolCall(
                name=normalized_tool_name,
                args=normalized_args,
                timestamp_sec=timestamp_sec,
                raw_entry=entry,
            ),
            issues,
        )
