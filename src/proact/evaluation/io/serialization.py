from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())
    if hasattr(value, "dict"):
        return to_serializable(value.dict())
    if hasattr(value, "to_dict"):
        return to_serializable(value.to_dict())
    if hasattr(value, "tolist"):
        return to_serializable(value.tolist())
    if hasattr(value, "item"):
        try:
            return to_serializable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return [to_serializable(item) for item in sorted(value)]
    return value


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_serializable(data)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_text(path: str | os.PathLike[str], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_evaluation_outputs(
    *,
    output_dir: str,
    config: Any,
    summary: dict[str, Any],
    execution_traces: list[Any],
    match_results: list[dict[str, Any]],
    skipped_matches: list[dict[str, Any]],
    save_execution_traces: bool = True,
    save_match_results: bool = True,
    save_skipped_matches: bool = True,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    config_path = Path(output_dir) / "config.json"
    summary_path = Path(output_dir) / "summary.json"
    manifest_path = Path(output_dir) / "manifest.json"

    write_json(config_path, config)
    write_json(summary_path, summary)

    manifest = {
        "config": str(config_path),
        "summary": str(summary_path),
    }
    if save_execution_traces:
        traces_path = Path(output_dir) / "execution_traces.json"
        write_json(traces_path, execution_traces)
        manifest["execution_traces"] = str(traces_path)
    if save_match_results:
        match_results_path = Path(output_dir) / "match_results.json"
        write_json(match_results_path, match_results)
        manifest["match_results"] = str(match_results_path)
    if save_skipped_matches:
        skipped_matches_path = Path(output_dir) / "skipped_matches.json"
        write_json(skipped_matches_path, skipped_matches)
        manifest["skipped_matches"] = str(skipped_matches_path)

    write_json(manifest_path, manifest)
    return manifest
