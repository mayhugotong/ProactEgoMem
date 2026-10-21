from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SampledFrame:
    array: np.ndarray          # H x W x C (RGB)
    frame_index: int           # index within this decode pass (0..)
    pts: Optional[int]         # raw presentation timestamp (stream units)
    time: float                # [when this frame actually occurs in the video. ]seconds (derived from PTS * time_base)
    chosen_for_time: float     # [the target time you asked for that this frame was picked to represent.] the target timestamp this frame matched (seconds)