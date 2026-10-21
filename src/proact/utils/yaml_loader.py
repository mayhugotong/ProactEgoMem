# yaml_reader.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import yaml


__all__ = [
    "read_yaml",
    "loads_yaml",
    "expand_env",
    "walk_map",
]


# Matches ${VAR_NAME} and replaces with os.environ["VAR_NAME"] (empty if missing by default).
_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _yaml_error_context(err: yaml.YAMLError, path: Path | None) -> str:
    """Build a concise error message with file/line/column when available."""
    where = f" in {path}" if path else ""
    if hasattr(err, "problem_mark") and err.problem_mark is not None:
        mark = err.problem_mark  # type: ignore[attr-defined]
        return f"YAML parse error{where} at line {mark.line + 1}, column {mark.column + 1}: {err}"
    return f"YAML parse error{where}: {err}"


def loads_yaml(text: str, *, expand_env_vars: bool = True) -> Any:
    """
    Parse YAML from a string.

    Args:
        text: YAML content.
        expand_env_vars: If True, expand ${ENV_VAR} inside all string values.

    Returns:
        Python object (dict/list/primitive).

    Raises:
        ValueError: On YAML parsing errors.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(_yaml_error_context(e, path=None)) from e

    if expand_env_vars:
        data = expand_env(data)
    return data


def read_yaml(
    path: str | os.PathLike,
    *,
    expand_env_vars: bool = True,
    expand_user: bool = True,
    must_exist: bool = True,
) -> Any:
    """
    Parse YAML from a file.

    Args:
        path: Path to a .yml/.yaml file.
        expand_env_vars: If True, expand ${ENV_VAR} in string values.
        expand_user: If True, expand '~' in strings and the input path itself.
        must_exist: If True, raise FileNotFoundError if the file is missing.

    Returns:
        Python object (dict/list/primitive).

    Raises:
        FileNotFoundError: If file is missing and must_exist=True.
        ValueError: On YAML parsing errors.
    """
    p = Path(path)
    if expand_user:
        p = Path(os.path.expanduser(str(p)))

    if not p.exists():
        if must_exist:
            raise FileNotFoundError(f"YAML file not found: {p}")
        return None

    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to read YAML file {p}: {e}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(_yaml_error_context(e, path=p)) from e

    if expand_env_vars or expand_user:
        # Apply string transforms across the entire structure.
        def _transform(s: str) -> str:
            if expand_user:
                s2 = os.path.expanduser(s)
            else:
                s2 = s
            if expand_env_vars:
                s2 = _ENV_VAR.sub(lambda m: os.environ.get(m.group(1), ""), s2)
            return s2

        data = walk_map(data, on_str=_transform)

    return data


def expand_env(value: Any) -> Any:
    """
    Recursively expand ${ENV_VAR} in all strings within 'value'.

    Missing variables resolve to empty string.

    Args:
        value: Arbitrary nested structure.

    Returns:
        Structure with environment variables expanded.
    """
    return walk_map(value, on_str=lambda s: _ENV_VAR.sub(lambda m: os.environ.get(m.group(1), ""), s))


def walk_map(
    value: Any,
    *,
    on_str: Callable[[str], str] | None = None,
    on_key: Callable[[str], str] | None = None,
) -> Any:
    """
    Recursively walk a nested structure (dict/list/tuples) and transform strings.

    Args:
        value: Input structure (dict/list/tuple/primitive).
        on_str: Optional function to transform string *values*.
        on_key: Optional function to transform dict string *keys*.

    Returns:
        Transformed structure.
    """
    if isinstance(value, str):
        return on_str(value) if on_str else value

    if isinstance(value, Mapping):
        out: MutableMapping[Any, Any] = {}.__class__()  # plain dict, preserves YAML order in 3.7+
        for k, v in value.items():
            new_k = on_key(k) if (on_key and isinstance(k, str)) else k
            out[new_k] = walk_map(v, on_str=on_str, on_key=on_key)
        return out

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        t = type(value)
        return t(walk_map(v, on_str=on_str, on_key=on_key) for v in value)

    return value

# -----------------------
# Yaml Writer
# -----------------------

def dumps_yaml(
    value: Any,
    *,
    sort_keys: bool = False,
    allow_unicode: bool = True,
) -> str:
    """
    Serialize a Python object to a YAML string.

    Args:
        value: Python object to serialize.
        sort_keys: If True, sort mapping keys in output.
        allow_unicode: If True, allow non-ASCII characters.

    Returns:
        YAML string.

    Raises:
        ValueError: On YAML serialization errors.
    """
    try:
        return yaml.safe_dump(
            value,
            sort_keys=sort_keys,
            allow_unicode=allow_unicode,
        )
    except yaml.YAMLError as e:
        raise ValueError(f"YAML dump error: {e}") from e


def write_yaml(
    value: Any,
    path: str | os.PathLike,
    *,
    sort_keys: bool = False,
    allow_unicode: bool = True,
    expand_user: bool = True,
    make_dirs: bool = True,
) -> None:
    """
    Serialize a Python object to a YAML file.

    Args:
        value: Python object to serialize.
        path: Target .yml/.yaml path.
        sort_keys: If True, sort mapping keys in output.
        allow_unicode: If True, allow non-ASCII characters.
        expand_user: If True, expand '~' in path.
        make_dirs: If True, create parent directories if missing.

    Raises:
        OSError: On write errors.
        ValueError: On YAML serialization errors.
    """
    p = Path(path)
    if expand_user:
        p = Path(os.path.expanduser(str(p)))

    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    text = dumps_yaml(value, sort_keys=sort_keys, allow_unicode=allow_unicode)

    try:
        p.write_text(text, encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to write YAML file {p}: {e}") from e



# -----------------------
# Minimal CLI (optional)
# -----------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Simple YAML reader")
    parser.add_argument("path", help="Path to YAML file")
    parser.add_argument("--no-env", action="store_true", help="Do not expand ${ENV_VAR}")
    parser.add_argument("--no-user", action="store_true", help="Do not expand ~")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    obj = read_yaml(
        args.path,
        expand_env_vars=not args.no_env,
        expand_user=not args.no_user,
    )
    if args.pretty:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(obj, ensure_ascii=False))
