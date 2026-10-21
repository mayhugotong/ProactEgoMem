from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike[str]) -> None:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid env line {line_number} in {env_path}: '{raw_line}'")
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()
