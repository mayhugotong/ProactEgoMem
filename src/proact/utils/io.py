# pip install av numpy imageio
# Requires FFmpeg (ffmpeg/ffprobe) available on your system.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import json

""" ######################################## LAOD BY PATH ######################################## """

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    
    return data

def load_txt(path):
    with open(path, "r") as f:
        main_prompt = f.read()
    
    return main_prompt

def save_txt(path: str, text: str) -> None:
    """Save a string to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


""" #################### ####################  SAVE JSON  #################### ####################"""
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def _clean_for_json(obj):
    """
    Recursively drop values that are not JSON-serializable.
    Works for dicts, lists, tuples, sets, and plain values.
    """
    # Dict: clean values, drop keys whose cleaned value isn't jsonable
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v_clean = _clean_for_json(v)
            if is_jsonable(v_clean):
                cleaned[k] = v_clean
        return cleaned

    # List / tuple / set: clean each element and drop non-jsonable ones
    elif isinstance(obj, (list, tuple, set)):
        cleaned_list = []
        for v in obj:
            v_clean = _clean_for_json(v)
            if is_jsonable(v_clean):
                cleaned_list.append(v_clean)
        return cleaned_list  # JSON only knows lists, so we return a list

    # Anything else: just return as-is, and the caller will check jsonability
    else:
        return obj

def save_json(save_path, data, overwrite=True, drop_unjsonable=True):
    """
    Save `data` as JSON.
    If drop_unjsonable=True, recursively removes any non-JSON-serializable
    values (e.g., numpy ndarrays) before dumping.
    """
    mode = "w" if overwrite else "x"  # 'x' = exclusive create; fails if exists

    if drop_unjsonable:
        data = _clean_for_json(data)

    with open(save_path, mode) as outfile:
        json.dump(data, outfile, indent=2)


##################################################### IMAGES ######################################################################
def save_numpy_as_jpg(array, to_path):
    from PIL import Image
    img = Image.fromarray(array.astype("uint8"))
    img.save(to_path, format="JPEG", quality=95)

    return to_path