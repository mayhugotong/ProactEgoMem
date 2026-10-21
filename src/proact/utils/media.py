# src/proact/utils/media.py
from __future__ import annotations
from pathlib import Path
import base64, mimetypes
import cv2
import functools
import shutil
import subprocess
import time
from proact.models.llm.base import ImageB64Part

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" ######################################## IMAGE PROCESSING ######################################## """
def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a HxWxC frame to exact (width, height).
    OpenCV expects dsize=(width, height).
    """
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame

    # Interpolation choice:
    # - INTER_AREA is usually best for downscaling because it reduces aliasing
    #   by effectively averaging pixel regions when shrinking the image.
    # - INTER_LINEAR is a good default for upscaling and for mixed resize
    #   (one dimension up, the other down), where AREA isn't clearly beneficial.
    interp = cv2.INTER_AREA if (width < w and height < h) else cv2.INTER_LINEAR

    return cv2.resize(frame, (width, height), interpolation=interp)


def _resolve_ffmpeg_path(ffmpeg_path: str | None = None) -> str | None:
    configured = ffmpeg_path or os.environ.get("PROACT_FFMPEG_PATH")
    if configured:
        expanded = os.path.expandvars(os.path.expanduser(configured))
        resolved = shutil.which(expanded)
        if resolved is None:
            raise RuntimeError(
                f"Configured ffmpeg binary does not exist or is not executable: {expanded}"
            )
        return resolved
    return shutil.which("ffmpeg")


@functools.lru_cache(maxsize=None)
def _ffmpeg_cuda_hwaccel_available(ffmpeg_path: str | None = None) -> bool:
    if os.environ.get("PROACT_DISABLE_GPU_FRAME_SAMPLING", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    resolved = _resolve_ffmpeg_path(ffmpeg_path)
    if resolved is None:
        return False
    try:
        result = subprocess.run(
            [resolved, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return "cuda" in (result.stdout or "").lower()


@functools.lru_cache(maxsize=None)
def _ffmpeg_filter_available(filter_name: str, ffmpeg_path: str | None = None) -> bool:
    resolved = _resolve_ffmpeg_path(ffmpeg_path)
    if resolved is None:
        return False
    try:
        result = subprocess.run(
            [resolved, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return any(
        line.split()[1:2] == [filter_name]
        for line in (result.stdout or "").splitlines()
        if line.strip()
    )


def _sample_frames_at_fps_ffmpeg_gpu(
    *,
    path: str,
    fps: float,
    S0_sec: float,
    S1_sec: float,
    include_endpoints: bool,
    convert_to_rgb: bool,
    resize_width: int | None,
    resize_height: int | None,
    ffmpeg_path: str | None = None,
) -> List[SampledFrame]:
    if not convert_to_rgb:
        raise RuntimeError("GPU FPS sampling currently requires convert_to_rgb=True.")
    resolved_ffmpeg = _resolve_ffmpeg_path(ffmpeg_path)
    if resolved_ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH and no ffmpeg_path was configured.")

    container = av.open(path)
    vstreams = [s for s in container.streams if s.type == "video"]
    if not vstreams:
        container.close()
        raise ValueError("No video stream found.")
    vstream = vstreams[0]
    src_width = int(vstream.width or 0)
    src_height = int(vstream.height or 0)
    container.close()

    width = int(resize_width or src_width or 0)
    height = int(resize_height or src_height or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine output frame size for GPU FPS sampling.")

    duration = float(S1_sec - S0_sec)
    if duration < 0:
        raise ValueError(f"Invalid range: S0={S0_sec} > S1={S1_sec}")

    N = max(1, int(math.floor(duration * fps)) + 1)
    targets = _make_targets_seconds(N, S0_sec, S1_sec, include_endpoints=include_endpoints)
    if not targets:
        return []

    gpu_resize = bool(
        resize_width and resize_height and _ffmpeg_filter_available("scale_cuda", resolved_ffmpeg)
    )
    vf_parts = []
    if gpu_resize:
        vf_parts.append(f"scale_cuda=w={width}:h={height}:format=nv12")
    vf_parts.extend(["hwdownload", "format=nv12", f"fps={fps}"])
    if resize_width and resize_height and not gpu_resize:
        vf_parts.append(f"scale={width}:{height}:flags=area")
    vf_parts.append("format=rgb24")
    vf = ",".join(vf_parts)

    cmd = [
        resolved_ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-ss",
        f"{S0_sec:.6f}",
        "-i",
        path,
        "-t",
        f"{duration:.6f}",
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg GPU FPS sampling failed: {stderr.strip()}")

    frame_size = width * height * 3
    if frame_size <= 0:
        raise RuntimeError("Invalid frame size for GPU FPS sampling.")
    payload = result.stdout
    if not payload:
        raise RuntimeError("ffmpeg GPU FPS sampling returned no frames.")
    if len(payload) % frame_size != 0:
        raise RuntimeError(
            f"ffmpeg GPU FPS sampling returned {len(payload)} bytes, which is not divisible by frame_size={frame_size}."
        )

    frame_count = len(payload) // frame_size
    expected_count = len(targets)
    # The decoded frame count may differ slightly from the target grid (ffmpeg's
    # fps filter is end-exclusive; boundary rounding varies per clip). The grid
    # timestamps are authoritative: accept what ffmpeg returned and align it to
    # the grid — never raise or fall back over a count mismatch.
    if frame_count < expected_count:
        logger.info(
            "FFmpeg FPS sampling returned %d/%d frames (terminal window "
            "rounding); accepting the decoded frames on the target grid.",
            frame_count,
            expected_count,
        )
        targets = targets[:frame_count]
    elif frame_count > expected_count:
        logger.info(
            "FFmpeg FPS sampling returned %d frames for %d targets; keeping "
            "the first %d on the target grid.",
            frame_count,
            expected_count,
            expected_count,
        )
        frame_count = expected_count

    arrays = np.frombuffer(payload, dtype=np.uint8)[: frame_count * frame_size].reshape(
        frame_count, height, width, 3
    )
    return [
        SampledFrame(
            array=arrays[idx].copy(),
            frame_index=idx,
            pts=None,
            time=float(target_time),
            chosen_for_time=float(target_time),
        )
        for idx, target_time in enumerate(targets)
    ]

""" ######################################## VIDEO PROCESSING ######################################## """
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import bisect
import math
import os
import av  # PyAV (FFmpeg bindings)
import numpy as np
from proact.config.schemas.video import SampledFrame

def _round_time_str(t: float, ndp: int = 2) -> str:
    """Decimal-rounded time string (e.g., 15.4321 -> '15.43') using ROUND_HALF_UP."""
    q = Decimal(10) ** -ndp
    return str(Decimal(str(t)).quantize(q, rounding=ROUND_HALF_UP))

def _get_stream_time_bounds(vstream) -> Tuple[float, float]:
    """Return (t_min, t_max) in seconds (best-effort)."""
    tb = float(vstream.time_base)
    if vstream.duration is not None:
        return 0.0, vstream.duration * tb
    return 0.0, float("inf")

def _make_targets_seconds(N: int, S0: float, S1: float, include_endpoints: bool) -> List[float]:
    if N <= 0:
        return []
    if S1 <= S0:
        return [S0] * N
    if include_endpoints:
        if N == 1:
            return [S0]
        step = (S1 - S0) / (N - 1)
        return [S0 + i * step for i in range(N)]
    else:
        step = (S1 - S0) / N
        return [S0 + (i + 0.5) * step for i in range(N)]

def _seconds_from_frames(S0_frames: int, S1_frames: int, avg_rate: float) -> Tuple[float, float]:
    if avg_rate <= 0:
        raise ValueError("Average frame rate unknown/invalid; cannot convert frame bounds to seconds.")
    return S0_frames / avg_rate, S1_frames / avg_rate

def sample_uniform_frames(
    path: str,
    N: int,
    S0: float | int | None = None,
    S1: float | int | None = None,
    unit: Literal["seconds", "frames"] = "seconds",
    include_endpoints: bool = False,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
) -> List[SampledFrame]:
    """
    Sample N frames uniformly in [S0, S1) by default; or [S0, S1] if include_endpoints=True.
    Uses nearest-by-timestamp selection → robust for CFR/VFR.

    Args
    ----
    unit:
      - "seconds" (recommended)
      - "frames" (uses average_rate to convert; exact for CFR, approx for VFR)
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return []

    container = av.open(path)
    vstreams = [s for s in container.streams if s.type == "video"]
    if not vstreams:
        raise ValueError("No video stream found.")
    vstream = vstreams[0]

    # Bounds
    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if not math.isfinite(t_max_stream) and container.duration is not None:
        t_max_stream = container.duration / av.time_base

    # Range → seconds
    if unit == "seconds":
        S0_sec = float(S0) if S0 is not None else t_min_stream
        S1_sec = float(S1) if S1 is not None else t_max_stream
    elif unit == "frames":
        rate = vstream.average_rate or getattr(vstream, "guessed_rate", None)
        if rate is None:
            raise ValueError("average_rate/guessed_rate unavailable; cannot convert frame indices.")
        avg = float(rate)
        s0f = int(S0) if S0 is not None else 0
        if S1 is None:
            if math.isfinite(t_max_stream):
                S1 = int(round(t_max_stream * avg))
            else:
                raise ValueError("Cannot infer S1 (frames) without known duration.")
        s1f = int(S1)
        S0_sec, S1_sec = _seconds_from_frames(s0f, s1f, avg)
    else:
        raise ValueError("unit must be 'seconds' or 'frames'")

    if not math.isfinite(t_max_stream):
        raise ValueError("Cannot determine video duration; provide S1 explicitly in seconds.")

    # Clamp and targets
    S0_sec = max(t_min_stream, S0_sec)
    S1_sec = min(t_max_stream, S1_sec)
    targets = _make_targets_seconds(N, S0_sec, S1_sec, include_endpoints=include_endpoints)

    # Seek near start for speed
    if precise_seek and targets:
        seek_pts = int(max(S0_sec - 2.0, 0.0) / vstream.time_base)
        try:
            container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
        except av.AVError:
            pass

    # Decode slightly wider window
    end_pad = max(0.5, (S1_sec - S0_sec) * 0.05)
    frames_times: List[float] = []
    frames_store: List[tuple[np.ndarray, int, Optional[int], float]] = []
    decode_idx = 0
    stop = False

    for packet in container.demux(vstream):
        for frame in packet.decode():
            if frame.pts is None or frame.time is None:
                continue
            t = float(frame.time)
            if t < (S0_sec - end_pad):
                continue
            if t > (S1_sec + end_pad):
                stop = True
                break
            img = frame.to_ndarray(format="rgb24") if convert_to_rgb else frame.to_ndarray()
            frames_store.append((img, decode_idx, frame.pts, t))
            frames_times.append(t)
            decode_idx += 1
        if stop:
            break

    if not frames_store:
        raise RuntimeError("No decodable frames found in the requested interval.")

    # Nearest-by-timestamp selection
    chosen: List[SampledFrame] = []
    used_indices = set()
    for ti in targets:
        j = bisect.bisect_left(frames_times, ti)
        candidates = []
        if j > 0: candidates.append(j - 1)
        if j < len(frames_times): candidates.append(j)
        best, best_dt = None, float("inf")
        for c in candidates:
            if unique and c in used_indices:
                continue
            dt = abs(frames_times[c] - ti)
            if dt < best_dt:
                best_dt, best = dt, c
        if best is None:
            for c in candidates:
                dt = abs(frames_times[c] - ti)
                if dt < best_dt:
                    best_dt, best = dt, c
        used_indices.add(best)
        arr, fidx, pts, ft = frames_store[best]
        chosen.append(SampledFrame(array=arr, frame_index=fidx, pts=pts, time=ft, chosen_for_time=ti))

    return chosen

def sample_uniform_frames_memory_friendly(
    path: str,
    N: int,
    S0: float | int | None = None,
    S1: float | int | None = None,
    unit: Literal["seconds", "frames"] = "seconds",
    include_endpoints: bool = False,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
) -> List[SampledFrame]:
    """
    Memory-friendly version of sample_uniform_frames.

    **FIXED:** Uses Presentation Time Stamp (PTS) to reliably link frames
    between Pass 1 (time scan) and Pass 2 (decode array), solving the index mismatch issue.
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return []

    # --- Setup and Bounds Calculation (Unmodified) ---
    container = av.open(path)
    vstreams = [s for s in container.streams if s.type == "video"]
    if not vstreams:
        container.close()
        raise ValueError("No video stream found.")
    vstream = vstreams[0]

    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if (not math.isfinite(t_max_stream)) and (container.duration is not None):
        t_max_stream = container.duration / av.time_base

    if unit == "seconds":
        S0_sec = float(S0) if S0 is not None else t_min_stream
        S1_sec = float(S1) if S1 is not None else t_max_stream
    elif unit == "frames":
        rate = vstream.average_rate or getattr(vstream, "guessed_rate", None)
        if rate is None:
            container.close()
            raise ValueError("average_rate/guessed_rate unavailable; cannot convert frame indices.")
        avg = float(rate)
        s0f = int(S0) if S0 is not None else 0
        if S1 is None:
            if math.isfinite(t_max_stream):
                S1 = int(round(t_max_stream * avg))
            else:
                container.close()
                raise ValueError("Cannot infer S1 (frames) without known duration.")
        s1f = int(S1)
        S0_sec, S1_sec = _seconds_from_frames(s0f, s1f, avg)
    else:
        container.close()
        raise ValueError("unit must be 'seconds' or 'frames'")

    if not math.isfinite(t_max_stream):
        container.close()
        raise ValueError("Cannot determine video duration; provide S1 explicitly in seconds.")

    S0_sec = max(t_min_stream, S0_sec)
    S1_sec = min(t_max_stream, S1_sec)
    targets = _make_targets_seconds(N, S0_sec, S1_sec, include_endpoints=include_endpoints)
    if not targets:
        container.close()
        return []
    end_pad = max(0.5, (S1_sec - S0_sec) * 0.05)

    #logging.info("[sample_uniform_frames] path=%s N=%d S0_sec=%.3f S1_sec=%.3f dur=%.3f", path, N, S0_sec, S1_sec, (S1_sec - S0_sec))


    # ---------- PASS 1: collect times, PTS, and window_idx ----------
    if precise_seek and targets:
        seek_pts = int(max(S0_sec - 2.0, 0.0) / vstream.time_base)
        try:
            container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
        except av.AVError:
            pass

    # Store tuples of (time, pts, window_idx)
    frames_meta: List[Tuple[float, int, int]] = []
    window_idx = 0  # index within filtered window
    stop = False

    for packet in container.demux(vstream):
        for frame in packet.decode():
            if frame.pts is None or frame.time is None:
                continue
            
            t = float(frame.time)
            
            if t < (S0_sec - end_pad):
                continue
            if t > (S1_sec + end_pad):
                stop = True
                break

            # Store all three identifiers
            frames_meta.append((t, frame.pts, window_idx))
            window_idx += 1

        if stop:
            break

    container.close()

    if not frames_meta:
        raise RuntimeError("No decodable frames found in the requested interval.")

    # Separate times for bisect
    frames_times = [meta[0] for meta in frames_meta]

    # --- choose indices (nearest-by-timestamp, tracking PTS and original window_idx) ---
    # Stores (chosen_window_idx_p1, chosen_pts, chosen_for_time)
    chosen_seq: List[Tuple[int, int, float]] = []
    used_indices: set[int] = set() # tracks window_idx for unique=True logic

    for ti in targets:
        j = bisect.bisect_left(frames_times, ti)
        candidates: List[int] = []
        if j > 0:
            candidates.append(j - 1)
        if j < len(frames_times):
            candidates.append(j)

        best: Optional[int] = None
        best_dt = float("inf")

        # Primary pass: respect uniqueness if possible.
        for c in candidates:
            if unique and frames_meta[c][2] in used_indices: # Check uniqueness against window_idx
                continue
            dt = abs(frames_times[c] - ti)
            if dt < best_dt:
                best_dt, best = dt, c

        # Fallback: ignore uniqueness constraint
        if best is None:
            for c in candidates:
                dt = abs(frames_times[c] - ti)
                if dt < best_dt:
                    best_dt, best = dt, c

        chosen_idx_meta = best
        
        # Extract the necessary metadata from the chosen index
        _, chosen_pts, chosen_window_idx = frames_meta[chosen_idx_meta]
        
        used_indices.add(chosen_window_idx) # Mark the frame as used based on window_idx
        
        # The sequence now tracks the frame by its unique PTS
        chosen_seq.append((chosen_window_idx, chosen_pts, ti))

    if not chosen_seq:
        raise RuntimeError("Could not select any frames.")

    # Map the unique PTS -> list of (window_idx_p1, target_time) that chose this frame
    pts_to_targets: dict[int, List[Tuple[int, float]]] = {}
    for window_idx_p1, pts, ti in chosen_seq:
        pts_to_targets.setdefault(pts, []).append((window_idx_p1, ti))

    needed_pts = set(pts_to_targets.keys())

    # ---------- PASS 2: decode ONLY chosen frames by PTS (FIXED) ----------
    container = av.open(path)
    vstreams2 = [s for s in container.streams if s.type == "video"]
    if not vstreams2:
        container.close()
        raise ValueError("No video stream found (second pass).")
    vstream2 = vstreams2[0]

    if precise_seek and targets:
        seek_pts = int(max(S0_sec - 2.0, 0.0) / vstream2.time_base)
        try:
            container.seek(seek_pts, stream=vstream2, any_frame=False, backward=True)
        except av.AVError:
            pass

    chosen: List[SampledFrame] = []
    stop = False
    
    # We do NOT use window_idx2 for matching, only for tracking the order 
    # if it were needed, but we rely entirely on PTS.
    
    for packet in container.demux(vstream2):
        for frame in packet.decode():
            if frame.pts is None or frame.time is None:
                continue
            
            t = float(frame.time)
            
            # Check the window bounds 
            if t < (S0_sec - end_pad):
                continue
            if t > (S1_sec + end_pad):
                stop = True
                break

            # Core fix: Match frame by its unique PTS
            if frame.pts in needed_pts:
                img = frame.to_ndarray(format="rgb24") if convert_to_rgb else frame.to_ndarray()
                pts = frame.pts

                # One SampledFrame per target that mapped to this PTS
                for window_idx_p1, ti in pts_to_targets[pts]:
                    chosen.append(
                        SampledFrame(
                            array=img,
                            frame_index=window_idx_p1,  # Use the index from Pass 1 for consistent semantics
                            pts=pts,
                            time=t,
                            chosen_for_time=ti,
                        )
                    )

                needed_pts.remove(pts)
                if not needed_pts:
                    stop = True
                    break

        if stop:
            break

    container.close()

    if not chosen:
        if needed_pts:
             raise RuntimeError("Failed to decode chosen frames in second pass. Decoded sequence may be inconsistent.")
        else:
            raise RuntimeError("Failed to decode chosen frames in second pass.")

    # Final step: sort the output sequence back into the original target order
    chosen.sort(key=lambda x: x.chosen_for_time)
    
    return chosen

import av
import math
import bisect
import logging
from typing import List, Tuple, Literal, Optional, Any

def sample_uniform_frames_memory_friendly_NEW(
    path: str,
    N: int,
    S0: float | int | None = None,
    S1: float | int | None = None,
    unit: Literal["seconds", "frames"] = "seconds",
    include_endpoints: bool = False,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
) -> List[Any]: # Replace Any with SampledFrame if defined elsewhere
    """
    Memory-friendly version of uniform frame sampling.
    Fixes the 'No decodable frames' error by ensuring the decoder is flushed
    and the time-window logic is strictly validated.
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return []

    # --- Setup and Bounds Calculation ---
    container = av.open(path)
    vstreams = [s for s in container.streams if s.type == "video"]
    if not vstreams:
        container.close()
        raise ValueError(f"No video stream found in {path}")
    vstream = vstreams[0]

    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if (not math.isfinite(t_max_stream)) and (container.duration is not None):
        t_max_stream = container.duration / av.time_base

    if unit == "seconds":
        S0_sec = float(S0) if S0 is not None else t_min_stream
        S1_sec = float(S1) if S1 is not None else t_max_stream
    elif unit == "frames":
        rate = vstream.average_rate or getattr(vstream, "guessed_rate", None)
        if rate is None:
            container.close()
            raise ValueError("average_rate unavailable; cannot convert frame indices.")
        avg = float(rate)
        s0f = int(S0) if S0 is not None else 0
        if S1 is None:
            if math.isfinite(t_max_stream):
                S1 = int(round(t_max_stream * avg))
            else:
                container.close()
                raise ValueError("Cannot infer S1 without known duration.")
        s1f = int(S1)
        S0_sec, S1_sec = _seconds_from_frames(s0f, s1f, avg)
    else:
        container.close()
        raise ValueError("unit must be 'seconds' or 'frames'")

    # --- Safety Clamping ---
    # Ensure S0 never exceeds S1, and both are within stream bounds
    S0_sec = max(t_min_stream, min(S0_sec, t_max_stream))
    S1_sec = max(S0_sec, min(S1_sec, t_max_stream))

    targets = _make_targets_seconds(N, S0_sec, S1_sec, include_endpoints=include_endpoints)
    if not targets:
        container.close()
        return []

    # Pad the window slightly to ensure we capture frames near edges
    end_pad = max(0.5, (S1_sec - S0_sec) * 0.05) if S1_sec > S0_sec else 0.5

    # ---------- PASS 1: Indexing (Metadata Only) ----------
    if precise_seek:
        # Seek slightly before S0 to ensure we don't skip the first frame
        seek_pts = int(max(S0_sec - 1.0, 0.0) / vstream.time_base)
        try:
            container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
        except av.AVError:
            pass

    frames_meta: List[Tuple[float, int, int]] = []
    window_idx = 0
    
    # Use container.decode() to automatically handle decoder flushing
    for frame in container.decode(vstream):
        if frame.pts is None or frame.time is None:
            continue
        
        t = float(frame.time)
        if t < (S0_sec - end_pad):
            continue
        if t > (S1_sec + end_pad):
            break

        frames_meta.append((t, frame.pts, window_idx))
        window_idx += 1

    container.close()

    if not frames_meta:
        # Fallback: if video is extremely short, try to get anything available
        raise RuntimeError(f"No decodable frames found in {path} between {S0_sec}s and {S1_sec}s.")

    # --- Nearest-Neighbor Selection ---
    frames_times = [meta[0] for meta in frames_meta]
    chosen_seq: List[Tuple[int, int, float]] = []
    used_indices: set[int] = set()

    for ti in targets:
        j = bisect.bisect_left(frames_times, ti)
        candidates = [c for c in [j - 1, j] if 0 <= c < len(frames_times)]
        
        best = None
        best_dt = float("inf")

        # Try to maintain uniqueness
        for c in candidates:
            if unique and frames_meta[c][2] in used_indices:
                continue
            dt = abs(frames_times[c] - ti)
            if dt < best_dt:
                best_dt, best = dt, c

        # Fallback if uniqueness is impossible
        if best is None:
            for c in candidates:
                dt = abs(frames_times[c] - ti)
                if dt < best_dt:
                    best_dt, best = dt, c

        _, chosen_pts, chosen_w_idx = frames_meta[best]
        used_indices.add(chosen_w_idx)
        chosen_seq.append((chosen_w_idx, chosen_pts, ti))

    # Map PTS -> List of target times (handles case where one frame is used for multiple targets)
    pts_to_targets: dict[int, List[Tuple[int, float]]] = {}
    for w_idx, pts, ti in chosen_seq:
        pts_to_targets.setdefault(pts, []).append((w_idx, ti))

    needed_pts = set(pts_to_targets.keys())

    # ---------- PASS 2: Extraction (Pixel Data) ----------
    container = av.open(path)
    vstream2 = [s for s in container.streams if s.type == "video"][0]

    if precise_seek:
        seek_pts = int(max(S0_sec - 1.0, 0.0) / vstream2.time_base)
        try:
            container.seek(seek_pts, stream=vstream2, any_frame=False, backward=True)
        except av.AVError:
            pass

    chosen: List[SampledFrame] = []
    
    for frame in container.decode(vstream2):
        if frame.pts in needed_pts:
            img = frame.to_ndarray(format="rgb24") if convert_to_rgb else frame.to_ndarray()
            
            for w_idx_p1, ti in pts_to_targets[frame.pts]:
                chosen.append(
                    SampledFrame(
                        array=img,
                        frame_index=w_idx_p1,
                        pts=frame.pts,
                        time=float(frame.time),
                        chosen_for_time=ti,
                    )
                )
            
            needed_pts.remove(frame.pts)
            if not needed_pts:
                break

    container.close()

    if not chosen:
        raise RuntimeError("Failed to decode any chosen frames in second pass.")

    # Restore original target order
    chosen.sort(key=lambda x: x.chosen_for_time)
    return chosen



def sample_frames_at_fps(
    path: str,
    fps: float,
    S0: float | None = None,
    S1: float | None = None,
    include_endpoints: bool = True,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
    resize_width: int | None = None,
    resize_height: int | None = None,
    backend: Literal["ffmpeg-cuda", "pyav-cpu"] = "ffmpeg-cuda",
    ffmpeg_path: str | None = None,
) -> List[SampledFrame]:
    """Sample frames at a fixed FPS and return only the frames."""
    frames, _ = sample_frames_at_fps_with_metadata(
        path=path, fps=fps, S0=S0, S1=S1, include_endpoints=include_endpoints,
        unique=unique, convert_to_rgb=convert_to_rgb, precise_seek=precise_seek,
        resize_width=resize_width, resize_height=resize_height, backend=backend,
        ffmpeg_path=ffmpeg_path,
    )
    return frames


def sample_frames_at_fps_with_metadata(
    path: str,
    fps: float,
    S0: float | None = None,
    S1: float | None = None,
    include_endpoints: bool = True,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
    resize_width: int | None = None,
    resize_height: int | None = None,
    backend: Literal["ffmpeg-cuda", "pyav-cpu"] = "ffmpeg-cuda",
    ffmpeg_path: str | None = None,
) -> tuple[List[SampledFrame], dict]:
    """Sample frames and report the decoder/backend that actually ran.

    Backend selection is EXPLICIT: "ffmpeg-cuda" (default) fails loudly when
    CUDA is unavailable or decoding errors; "pyav-cpu" must be requested
    deliberately. There is no auto-detection and no silent fallback.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if backend not in {"ffmpeg-cuda", "pyav-cpu"}:
        raise ValueError(f"Unsupported frame sampling backend: {backend}")

    container = av.open(path)
    vstreams = [stream for stream in container.streams if stream.type == "video"]
    if not vstreams:
        container.close()
        raise ValueError("No video stream found.")
    vstream = vstreams[0]
    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if not math.isfinite(t_max_stream) and container.duration is not None:
        t_max_stream = container.duration / av.time_base
    if not math.isfinite(t_max_stream):
        container.close()
        raise ValueError("Cannot determine video duration; provide S1 explicitly in seconds.")

    S0_sec = max(t_min_stream, float(S0) if S0 is not None else t_min_stream)
    S1_sec = min(t_max_stream, float(S1) if S1 is not None else t_max_stream)
    duration = S1_sec - S0_sec
    if duration < 0:
        container.close()
        raise ValueError(f"Invalid range: S0={S0_sec} > S1={S1_sec}")
    N = max(1, int(math.floor(duration * fps)) + 1)
    container.close()

    if backend == "ffmpeg-cuda":
        resolved_ffmpeg = _resolve_ffmpeg_path(ffmpeg_path)
        cuda_available = _ffmpeg_cuda_hwaccel_available(resolved_ffmpeg)
        logger.info(
            "Frame sampling backend selection: path=%s requested_backend=%s "
            "ffmpeg_path=%s cuda_advertised=%s",
            path,
            backend,
            resolved_ffmpeg,
            cuda_available,
        )
        if not cuda_available:
            raise RuntimeError(
                "GPU frame sampling requested (backend=ffmpeg-cuda) but this "
                "ffmpeg build does not advertise the cuda hardware accelerator. "
                "Run on a GPU node or set frame_sampling.backend=pyav-cpu explicitly."
            )
        gpu_resize = bool(
            resize_width and resize_height and _ffmpeg_filter_available("scale_cuda", resolved_ffmpeg)
        )
        logger.info(
            "Frame sampling backend running: path=%s selected_backend=ffmpeg-cuda "
            "processing_path=%s",
            path,
            "cuda-decode+cuda-scale+cpu-fps" if gpu_resize else "cuda-decode+cpu-fps-scale",
        )
        started_at = time.perf_counter()
        # Errors propagate: there is intentionally no CPU fallback.
        frames = _sample_frames_at_fps_ffmpeg_gpu(
            path=path, fps=float(fps), S0_sec=S0_sec, S1_sec=S1_sec,
            include_endpoints=include_endpoints, convert_to_rgb=convert_to_rgb,
            resize_width=resize_width, resize_height=resize_height,
            ffmpeg_path=resolved_ffmpeg,
        )
        return frames, {
            "requested_backend": backend,
            "selected_backend": "ffmpeg-cuda",
            "decoder": "cuda",
            "ffmpeg_path": resolved_ffmpeg,
            "processing_path": "cuda-decode+cuda-scale+cpu-fps" if gpu_resize else "cuda-decode+cpu-fps-scale",
            "gpu_resize": gpu_resize,
            "elapsed_sec": time.perf_counter() - started_at,
        }

    logger.info(
        "Frame sampling backend running: path=%s selected_backend=pyav-cpu "
        "processing_path=pyav-two-pass",
        path,
    )
    started_at = time.perf_counter()
    frames = sample_uniform_frames_memory_friendly(
        path=path, N=N, S0=S0_sec, S1=S1_sec, unit="seconds",
        include_endpoints=include_endpoints, unique=unique,
        convert_to_rgb=convert_to_rgb, precise_seek=precise_seek,
    )
    return frames, {
        "requested_backend": backend,
        "selected_backend": "pyav-cpu",
        "decoder": "pyav",
        "ffmpeg_path": None,
        "processing_path": "pyav-two-pass",
        "gpu_resize": False,
        "elapsed_sec": time.perf_counter() - started_at,
    }


from typing import List, Literal, Optional
import math
import numpy as np


def save_sampled_frames_as_images(
    sampled: List[SampledFrame],
    out_dir: str,
    prefix: str = "frame",
    ndp: int = 2,
    filename_template: Optional[str] = None,
) -> List[str]:
    """
    Save sampled frames as PNGs to a chosen folder.

    filename_template fields:
      {index}, {time_display}, {chosen_time}, {target_time}, {pts}
      e.g., "{index:03d}_{time_display}.png"
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        from imageio import imwrite
    except ImportError:
        raise ImportError("Please `pip install imageio` to use save_sampled_frames_as_images.")

    written_paths: List[str] = []
    for i, f in enumerate(sampled):
        tdisp = _round_time_str(f.time, ndp)
        if filename_template:
            fname = filename_template.format(
                index=i,
                time_display=tdisp,
                chosen_time=f.time,
                target_time=f.chosen_for_time,
                pts=f.pts,
            )
        else:
            fname = f"{prefix}_{i:03d}_{tdisp}.png"
        path = os.path.join(out_dir, fname)
        imwrite(path, f.array)
        written_paths.append(os.path.abspath(path))
    return written_paths

from typing import Literal

def sample_single_frame(
    path: str,
    t: float | int,
    unit: Literal["seconds", "frames"] = "seconds",
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
) -> SampledFrame:
    """
    Sample the single frame closest to a target time.

    Args
    ----
    path:
        Video file path.
    t:
        Target time:
          - If unit == "seconds": time in seconds (float or int).
          - If unit == "frames": frame index (int).
    unit:
        "seconds" or "frames" (same semantics as sample_uniform_frames).
    convert_to_rgb:
        Passed through to sample_uniform_frames.
    precise_seek:
        Passed through to sample_uniform_frames.

    Returns
    -------
    SampledFrame
        The frame whose timestamp is closest to t (using your
        nearest-by-timestamp selection).
    """
    frames = sample_uniform_frames(
        path=path,
        N=1,
        S0=t,
        S1=t,
        unit=unit,
        include_endpoints=True,  # degenerate [t, t] → single target at t
        unique=True,
        convert_to_rgb=convert_to_rgb,
        precise_seek=precise_seek,
    )
    # sample_uniform_frames(N=1, S0=t, S1=t) will either raise or return [frame]
    return frames[0]



""" ######################################## VIDEO SUBTITLE GENERATION #############################################"""


"""if __name__ == "__main__":
    # NOTE: SAMPLE CALL
    video_path = "03d4c383-b80e-4095-9328-c964f2803a26.mp4"

    # Sample 12 frames in [21,25] including both endpoints
    sampled = sample_uniform_frames(
        path=video_path,
        N=12,
        S0=21,
        S1=25,
        unit="seconds",
        include_endpoints=True,   # closed interval [S0, S1]
        unique=True,
        convert_to_rgb=True,
        precise_seek=True,
    )

    # Save where you want, with your naming
    written = save_sampled_frames_as_images(
        sampled,
        out_dir="runs/vid01/21to25",
        prefix="sample",
        filename_template="{index:03d}_{time_display}.png",  # e.g., 000_21.00.png
        ndp=2,
    )

    # You still have precise mapping in the SampledFrame list:
    for sf in sampled:
        print(
            f"index={sf.frame_index}  target={sf.chosen_for_time:.6f}s  "
            f"chosen={sf.time:.6f}s  pts={sf.pts}"
        )

    print("First written file:", written[0] if written else None)"""

""" ***************** STREAMING ***************** """
import bisect
import math
import av
from collections import deque, defaultdict
from typing import List, Tuple, Iterator, Optional, Deque
import numpy as np

# Adjust imports to match your project structure
#from ..config.schemas.video import SampledFrame
#from .media import _get_stream_time_bounds, _make_targets_seconds, _seconds_from_frames

""" ######################################## SUB-FUNCTIONS ######################################## """

def _scan_video_metadata(
    path: str,
    S0_sec: float,
    S1_sec: float,
    precise_seek: bool = True
) -> Tuple[List[float], List[int]]:
    """
    PASS 1: Scans the video to get precise (Time, PTS) pairs.
    Memory Friendly: Does NOT store image data, only floats/ints.
    """
    container = av.open(path)
    vstream = container.streams.video[0]

    # Heuristic for padding
    end_pad = max(0.5, (S1_sec - S0_sec) * 0.05)

    # Seek
    if precise_seek:
        seek_pts = int(max(S0_sec - 2.0, 0.0) / vstream.time_base)
        try:
            container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
        except av.AVError:
            pass

    times: List[float] = []
    ptss: List[int] = []
    
    stop_threshold = S1_sec + end_pad

    for packet in container.demux(vstream):
        for frame in packet.decode():
            if frame.pts is None or frame.time is None:
                continue
            
            t = float(frame.time)
            
            if t < (S0_sec - end_pad):
                continue
            if t > stop_threshold:
                # We can stop early
                container.close()
                return times, ptss

            times.append(t)
            ptss.append(frame.pts)

    container.close()
    return times, ptss


def _map_targets_to_pts(
    target_times: List[float],
    available_times: List[float],
    available_ptss: List[int],
    unique: bool = True
) -> List[Tuple[float, int]]:
    """
    Performs the "Global Nearest Neighbor" logic.
    Returns a list of (target_time, best_matching_pts).
    """
    chosen_pairs: List[Tuple[float, int]] = []
    used_pts = set()

    for ti in target_times:
        # Find insertion point
        j = bisect.bisect_left(available_times, ti)
        
        # Check neighbors (left and right)
        candidates = []
        if j > 0: 
            candidates.append(j - 1)
        if j < len(available_times): 
            candidates.append(j)

        best_idx = None
        best_dt = float("inf")

        # Logic matches sample_uniform_frames_memory_friendly exactly
        # 1. Try to find unused frame
        for c in candidates:
            pts = available_ptss[c]
            if unique and (pts in used_pts):
                continue
            dt = abs(available_times[c] - ti)
            if dt < best_dt:
                best_dt = dt
                best_idx = c

        # 2. Fallback: reuse frame if necessary
        if best_idx is None:
            for c in candidates:
                dt = abs(available_times[c] - ti)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = c

        if best_idx is not None:
            chosen_pts = available_ptss[best_idx]
            used_pts.add(chosen_pts)
            chosen_pairs.append((ti, chosen_pts))

    return chosen_pairs


def _generate_decoded_stream(
    path: str,
    schedule: List[Tuple[float, int]], # List of (target_time, pts_to_fetch)
    convert_to_rgb: bool = True
) -> Iterator[SampledFrame]:
    """
    PASS 2 (Generator): 
    Opens the video again. Decodes ONLY the needed PTS.
    Yields SampledFrames in the order requested by 'schedule'.
    """
    if not schedule:
        return

    # 1. Identify unique PTS needed to avoid decoding same frame twice
    unique_needed_pts = set(pts for _, pts in schedule)
    
    # 2. Open Video
    container = av.open(path)
    vstream = container.streams.video[0]

    # seek to the earliest needed PTS (minus buffer)
    min_pts = min(unique_needed_pts)
    seek_pts = max(0, min_pts - int(1.0 / vstream.time_base * vstream.time_base)) # 1s buffer
    try:
        container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
    except av.AVError:
        pass

    # 3. Buffer for decoded images: { pts: np.ndarray }
    # We need this because the video decodes in stream order, 
    # but our schedule might request them in a slightly different order (rare) 
    # or multiple times.
    decoded_buffer = {}
    
    # Tracks how many times each PTS is still needed in the schedule
    pts_ref_count = defaultdict(int)
    for _, pts in schedule:
        pts_ref_count[pts] += 1

    schedule_idx = 0
    total_items = len(schedule)

    # 4. Stream Loop
    # We iterate packets. Inside, we check if we have the frame for schedule[schedule_idx]
    
    video_iter = container.demux(vstream)
    
    while schedule_idx < total_items:
        target_time, needed_pts = schedule[schedule_idx]

        # Check if we already have the frame buffered
        if needed_pts in decoded_buffer:
            # Yield result
            img = decoded_buffer[needed_pts]
            
            # We don't have the exact original decode time 't' here easily 
            # without storing it in Pass 1 or buffer. 
            # But usually for ML, the array and target_time are crucial.
            # If precise capture time is needed, we can store it in decoded_buffer too.
            yield SampledFrame(
                array=img,
                frame_index=schedule_idx, 
                pts=needed_pts,
                time=target_time, # Using target time as approx, or we could map back
                chosen_for_time=target_time
            )

            # Decrement ref count
            pts_ref_count[needed_pts] -= 1
            if pts_ref_count[needed_pts] <= 0:
                del decoded_buffer[needed_pts]
            
            schedule_idx += 1
            continue

        # If not in buffer, we must read from video until we find it
        try:
            packet = next(video_iter)
            for frame in packet.decode():
                if frame.pts is None: 
                    continue
                
                # Is this frame needed by ANYONE?
                if frame.pts in unique_needed_pts:
                    img = frame.to_ndarray(format="rgb24") if convert_to_rgb else frame.to_ndarray()
                    decoded_buffer[frame.pts] = img
                    # Note: We loop back to the 'while' start to check if this 
                    # satisfied the current need.
        except StopIteration:
            # Video ended but we still have schedule items?
            break

    container.close()


""" ######################################## MAIN GENERATOR ######################################## """

def precise_sliding_window_generator(
    path: str,
    S0: float | int | None,
    S1: float | int | None,
    fps: float,
    max_window_size: int,
    convert_to_rgb: bool = True,
    shape: Optional[Tuple[int, int]] = None, # (H, W)
) -> Iterator[List[SampledFrame]]:
    """
    High-Precision Sliding Window Generator.
    
    1. PASS 1 (Fast): Scans all frame timestamps/PTS (metadata only).
    2. CALC: Determines the globally optimal (Nearest Neighbor) PTS for every target step.
    3. PASS 2 (Stream): Decodes only the chosen frames and yields sliding windows.
    """
    
    # --- 1. Setup Targets ---
    # Convert FPS to count N, or just generate grid
    if fps <= 0: raise ValueError("FPS must be > 0")

    # --- resolve None to stream min/max (same semantics as your other funcs) ---
    container = av.open(path)
    vstream = container.streams.video[0]

    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if (not math.isfinite(t_max_stream)) and (container.duration is not None):
        t_max_stream = container.duration / av.time_base

    container.close()

    if not math.isfinite(t_max_stream):
        raise ValueError("Cannot determine video duration; provide S1 explicitly in seconds.")

    S0 = float(S0) if S0 is not None else t_min_stream
    S1 = float(S1) if S1 is not None else t_max_stream

    # clamp
    S0 = max(t_min_stream, S0)
    S1 = min(t_max_stream, S1)

    duration = S1 - S0
    if duration < 0: return # Empty
    
    # Generate the exact grid of times we want
    N = int(math.floor(duration * fps)) + 1
    target_times = [S0 + i * (1/fps) for i in range(N)]
    
    if not target_times:
        return

    # --- 2. Pass 1: Metadata Scan (Memory Efficient) ---
    # This reads the whole file structure but stores 0 images.
    avail_times, avail_ptss = _scan_video_metadata(path, S0, S1)
    
    if not avail_times:
        return

    # --- 3. Match Targets to PTS ---
    # This gives us the "Schedule": [(time=0.0, pts=1024), (time=0.5, pts=2048)...]
    schedule = _map_targets_to_pts(target_times, avail_times, avail_ptss, unique=True)
    
    # --- 4. Pass 2: Generator & Windowing ---
    window: Deque[SampledFrame] = deque(maxlen=max_window_size)
    
    # This generator yields SampledFrames one by one
    frame_stream = _generate_decoded_stream(path, schedule, convert_to_rgb)
    
    for frame in frame_stream:
        if shape is not None:
            H, W = shape
            # your helper expects (width, height)
            frame.array = resize_frame(frame.array, width=W, height=H)

        window.append(frame)
        # Yield a list copy of the current window
        yield list(window)


#### APPARENTLY TRUE STREAMING
from collections import deque
from typing import Iterator, List, Optional, Literal
import av
import math

def streaming_sliding_window_generator(
    path: str,
    S0: float | None,
    S1: float | None,
    fps: float,
    max_window_size: int,
    include_endpoints: bool = True,
    unique: bool = True,
    convert_to_rgb: bool = True,
    precise_seek: bool = True,
    seek_back_sec: float = 1.0,   # decode a little before S0 so prev exists
) -> Iterator[List[SampledFrame]]:
    """
    TRUE streaming:
      - single decode pass
      - yields windows as soon as targets are hit
      - O(1) memory aside from the deque window
    Tradeoff:
      - decodes frames between S0..S1 (can't "decode only chosen" without prescan or repeated seeks).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    step = 1.0 / fps

    container = av.open(path)
    vstream = container.streams.video[0]

    # Resolve bounds like your other funcs
    t_min_stream, t_max_stream = _get_stream_time_bounds(vstream)
    if (not math.isfinite(t_max_stream)) and (container.duration is not None):
        t_max_stream = container.duration / av.time_base
    if not math.isfinite(t_max_stream):
        container.close()
        raise ValueError("Cannot determine video duration; provide S1 explicitly.")

    S0_sec = float(S0) if S0 is not None else t_min_stream
    S1_sec = float(S1) if S1 is not None else t_max_stream

    S0_sec = max(t_min_stream, S0_sec)
    S1_sec = min(t_max_stream, S1_sec)
    if S1_sec < S0_sec:
        container.close()
        return

    # Initialize target time (grid generated on-the-fly)
    target = S0_sec if include_endpoints else (S0_sec + 0.5 * step)

    # Seek a bit before start so we have a "prev" frame when we reach first target
    if precise_seek:
        seek_to = max(S0_sec - seek_back_sec, 0.0)
        seek_pts = int(seek_to / vstream.time_base)
        try:
            container.seek(seek_pts, stream=vstream, any_frame=False, backward=True)
        except av.AVError:
            pass

    prev_frame = None            # PyAV VideoFrame
    prev_time: Optional[float] = None

    last_chosen_pts: Optional[int] = None
    out_index = 0

    window = deque(maxlen=max_window_size)

    def to_np(frm):
        return frm.to_ndarray(format="rgb24") if convert_to_rgb else frm.to_ndarray()

    # Decode in presentation order; yields start as soon as we hit target
    for frm in container.decode(vstream):
        if frm.pts is None or frm.time is None:
            continue

        t = float(frm.time)

        # Skip until we're close to start
        if t < (S0_sec - seek_back_sec):
            prev_frame = frm
            prev_time = t
            continue

        # Stop once we are past end and already produced all targets
        if t > S1_sec and target > S1_sec:
            break

        # When we cross target(s), decide using (prev, curr)
        while target <= S1_sec and t >= target:
            # Pick closest among prev and curr
            chosen = frm
            chosen_t = t

            if prev_frame is not None and prev_time is not None:
                dt_prev = abs(prev_time - target)
                dt_curr = abs(t - target)
                if dt_prev <= dt_curr:
                    chosen = prev_frame
                    chosen_t = prev_time

            # Uniqueness best-effort (same semantics as your fallback logic)
            if unique and chosen.pts is not None and last_chosen_pts is not None:
                if chosen.pts == last_chosen_pts:
                    # try the other candidate if available
                    other = frm if chosen is prev_frame else prev_frame
                    other_t = t if other is frm else prev_time
                    if other is not None and other.pts is not None and other.pts != last_chosen_pts:
                        chosen = other
                        chosen_t = float(other_t)

            img = to_np(chosen)
            pts = int(chosen.pts) if chosen.pts is not None else None

            sf = SampledFrame(
                array=img,
                frame_index=out_index,
                pts=pts,
                time=float(chosen_t),
                chosen_for_time=float(target),
            )
            out_index += 1
            if pts is not None:
                last_chosen_pts = pts

            window.append(sf)
            yield list(window)

            target += step

        prev_frame = frm
        prev_time = t

    # If video ended but there are remaining targets: emit last known frame
    while target <= S1_sec and prev_frame is not None and prev_time is not None:
        img = to_np(prev_frame)
        pts = int(prev_frame.pts) if prev_frame.pts is not None else None
        sf = SampledFrame(
            array=img,
            frame_index=out_index,
            pts=pts,
            time=float(prev_time),
            chosen_for_time=float(target),
        )
        out_index += 1
        window.append(sf)
        yield list(window)
        target += step

    container.close()


def sample_single_frame_safe(path: str, t: float, **kwargs) -> SampledFrame:
    container = av.open(path)
    vstream = container.streams.video[0]
    t_min, t_max = _get_stream_time_bounds(vstream)
    if (not math.isfinite(t_max)) and (container.duration is not None):
        t_max = container.duration / av.time_base
    container.close()

    if not math.isfinite(t_max):
        raise ValueError("Cannot determine duration; provide a valid in-range timestamp.")

    t = max(t_min, min(float(t), float(t_max)))
    return sample_uniform_frames_memory_friendly(
        path=path,
        N=1,
        S0=t,
        S1=t,
        unit="seconds",
        include_endpoints=True,
        unique=True,
        **kwargs,
    )[0]


import os
from shutil import which

import subprocess
import numpy as np

def frame_at(video, t):
    # 1) Video-Größe holen (einmal)
    w,h = map(int, subprocess.check_output([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0:s=x", video
    ]).decode().strip().split("x"))

    # 2) Hybrid seek: schnell + präzise, Frame als raw RGB in stdout
    m = 1.0
    fast = max(t - m, 0.0)
    acc  = t - fast
    raw = subprocess.check_output([
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss",f"{fast:.6f}","-i",video,"-ss",f"{acc:.6f}",
        "-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"
    ])

    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)

# usage:
# img = frame_at("input.mp4", 12.34)   # img.shape == (H,W,3), RGB


if __name__ == "__main__":
    path = "/path/to/proact-benchmark/data/test/cba13243-5ad3-4360-87df-6286867c473a.mp4"
    # --- FOR DEBUGGING ONLY ---
    OUTPUT_DIR = "/path/to/proact-benchmark/tests/img_stream"
    def save_frame(path: str, frame: SampledFrame, ndp: int = 4) -> str:
        """
        Saves a single SampledFrame to disk using a precise, descriptive filename.
        """
        # You'll need to ensure 'PIL' or 'Pillow' is installed: pip install Pillow
        try:
            from PIL import Image
        except ImportError:
            # Raise a clear error if the dependency is missing
            raise ImportError("The 'Pillow' library is required to save frames. Please install it with: pip install Pillow")

        # Create the descriptive filename based on target time and PTS
        tdisp = f"{frame.chosen_for_time:.{ndp}f}"
        fname = f"frame_{tdisp}s_pts{frame.pts}.png"
        
        # Ensure the directory exists
        Path(path).mkdir(parents=True, exist_ok=True)
        
        full_path = os.path.join(path, fname)
        
        # **PIL Conversion and Save**
        # The frame.array is a NumPy array (H x W x C). PIL handles the conversion.
        img = Image.fromarray(frame.array)
        img.save(full_path)
        
        return full_path
    
    # Precise sampling at 4 FPS between sec 10 and 15
    # Context window of 5 frames
    gen = precise_sliding_window_generator(
        path=path,
        S0=0,
        S1=60,
        fps=0.2,
        max_window_size=5
    )
    
    for i, window in enumerate(gen):
        print(f"Step {i}: Window size {len(window)}")
        # Verify precision
        last_frame = window[-1]
        # --- YOUR MINIMAL DEBUGGER ---
        print(f"Step {i:03d}: Window size {len(window)}")
        print(f"   Target: {last_frame.chosen_for_time:.4f}s | Chosen Time: {last_frame.time:.4f}s | PTS: {last_frame.pts}")
        
        # Use the new minimal function
        saved_path = save_frame(OUTPUT_DIR, last_frame) 
        print(f"   -> Saved: {os.path.basename(saved_path)}")
        # --- END MINIMAL DEBUGGER ---

    print("-" * 50)
    print("Finished stream.")

    print("NOW SERIOUSLY")
    path = "/path/to/proact-benchmark/data/test/cba13243-5ad3-4360-87df-6286867c473a.mp4"
    res = sample_frames_at_fps(
        path,
        S0=0,
        S1=1,
        fps=10
    )

    for frame in res:
        save_frame("/path/to/proact-benchmark/outputs/visuals", frame)
