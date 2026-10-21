""" ############################ MATH UTILS ############################ """

from __future__ import annotations

import math

def seconds_to_hhmmss(s: str | float, *, round_method: str = "nearest") -> str:
    x = float(s)
    if round_method == "floor":
        total = math.floor(x)
    elif round_method == "ceil":
        total = math.ceil(x)
    else:  # "nearest"
        total = int(round(x))

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

import re

def hhmmss_to_seconds(t: str) -> float:
    """
    Inverse of `seconds_to_hhmmss`.

    Accepts:
      - "HH:MM:SS"
      - "MM:SS"
      - "SS"
    Seconds may include a fractional part (e.g., "12:34:56.789" or "08.5").

    Returns total seconds as float (use int(...) if you want whole seconds).
    Raises ValueError on invalid input.
    """
    t = t.strip()
    if not t:
        raise ValueError("Empty time string.")

    # Split and allow 1–3 fields
    parts = t.split(":")
    if not (1 <= len(parts) <= 3):
        raise ValueError(f"Invalid time format: {t!r}")

    # Validate numeric (allow fractional on the last field only)
    num_re = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
    for i, p in enumerate(parts):
        p = p.strip()
        if i < len(parts) - 1:
            # hours/minutes must be integers
            if not p.isdigit():
                raise ValueError(f"Expected integer in position {i+1}: {p!r}")
        else:
            # seconds can be float
            if not num_re.match(p):
                raise ValueError(f"Invalid seconds field: {p!r}")

    # Map to h, m, s
    if len(parts) == 3:
        h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    elif len(parts) == 2:
        h = 0; m = int(parts[0]); s = float(parts[1])
    else:
        h = 0; m = 0; s = float(parts[0])

    if not (0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"Minutes/seconds out of range: m={m}, s={s}")

    return h * 3600 + m * 60 + s
