# src/proact/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os
import re
import yaml

# Factory: create_llm(provider: str, **cfg_kwargs) -> LLM
from ..models.llm.registry import create_llm

# ---------------------------
# Simple ${ENV_VAR} expansion
# ---------------------------
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _expand_env(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand_env(v) for v in x]
    if isinstance(x, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), x)
    return x

# ---------------------------
# Basic YAML helpers
# ---------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand_env(data)

def _select_profile(data: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    """
    If the YAML has {profiles: {...}}, pick one (or default_profile).
    Otherwise treat the whole YAML as the profile.
    """
    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        name = profile or data.get("default_profile")
        if not name and len(profiles) == 1:
            name = next(iter(profiles))
        if not name:
            raise ValueError(
                "No profile specified and no default_profile. "
                f"Available: {list(profiles)}"
            )
        if name not in profiles:
            raise KeyError(f"Profile '{name}' not found. Available: {list(profiles)}")
        return profiles[name]

    if "provider" not in data and "model" not in data and all(
        isinstance(value, dict) for value in data.values()
    ):
        name = profile or data.get("default_profile")
        if not name:
            if len(data) == 1:
                name = next(iter(data))
            else:
                raise ValueError(
                    "No profile specified for multi-profile config. "
                    f"Available: {list(data)}"
                )
        if name not in data:
            raise KeyError(f"Profile '{name}' not found. Available: {list(data)}")
        return data[name]
    return data

# ---------------------------
# Normalization (ultra-simple)
# ---------------------------
_ALLOWED_TOP_LEVEL = {
    "provider", "model",
    "api_base", "api_key",
    "request_timeout_s", "tps_limit", "rpm_limit",
    "cache_enabled",
    "params", "extra", "engine", "data"
}

def _normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal normalization:
    - Require 'provider' and 'model'.
    - Pass through only known top-level keys.
    - Assume 'params', 'extra', 'engine' are already well-formed (dicts).
    - No key hoisting, no deep merges, no surprises.
    """
    if "provider" not in raw or "model" not in raw:
        raise ValueError("Profile must include 'provider' and 'model'.")

    cfg: Dict[str, Any] = {}
    for k in _ALLOWED_TOP_LEVEL:
        if k in raw:
            cfg[k] = raw[k]

    # Basic type checks for the three dict sections
    for section in ("params", "extra", "engine", "data"):
        if section in cfg and cfg[section] is None:
            cfg[section] = {}
        if section in cfg and not isinstance(cfg[section], dict):
            raise TypeError(f"'{section}' must be a mapping if provided.")

    # Ensure required remain
    if "provider" not in cfg or "model" not in cfg:
        raise ValueError("Normalized config lost required 'provider' or 'model'.")

    # Provide defaults
    cfg.setdefault("params", {})
    cfg.setdefault("extra", {})
    cfg.setdefault("engine", {})
    cfg.setdefault("data", {})

    return cfg

# ---------------------------
# Public API
# ---------------------------
def load_profile(
    path: str | os.PathLike,
    profile: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a profile mapping from YAML and normalize to kwargs for registry.create_llm.
    - No smart merging/hoisting.
    - overrides: shallow, top-level replacement (including params/extra/engine).
    """
    p = Path(path)
    data = _read_yaml(p)
    selected = _select_profile(data, profile)
    cfg = _normalize_profile(dict(selected))  # shallow copy

    if overrides:
        # Replace top-level keys outright (including dict sections)
        for k, v in overrides.items():
            cfg[k] = v

        if "provider" not in cfg or "model" not in cfg:
            raise ValueError("Overrides removed required 'provider' or 'model' fields.")

        # Keep dict sections as dicts if present
        for section in ("params", "extra", "engine", "data"):
            if section in cfg and cfg[section] is None:
                cfg[section] = {}
            if section in cfg and not isinstance(cfg[section], dict):
                raise TypeError(f"Override for '{section}' must be a mapping.")

    return cfg

def build_llm_from_yaml(
    path: str | os.PathLike,
    profile: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Convenience: load + normalize + construct LLM via registry."""
    cfg = load_profile(path, profile, overrides)
    provider = cfg.pop("provider")
    return create_llm(provider=provider, **cfg)

__all__ = ["load_profile", "build_llm_from_yaml"]
