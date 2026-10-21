from proact.utils.math_utils import hhmmss_to_seconds, seconds_to_hhmmss
from proact.utils.io import load_json
from proact.data.datasets.utils.metadata import Metadata
import os
import logging
from pathlib import Path
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_narration_line(narration, timeformat="seconds"):
    # 1. Set conversion function to seconds to hhmmss or simply to Identity function to keep seconds
    conversion_fn = seconds_to_hhmmss if timeformat == "hhmmss" else lambda x: round(x, 3)

    return f"{conversion_fn(narration['timestamp_sec'])}: {narration['narration_text'].strip()}"


class Narrations:
    def __init__(self, annotation_root):
        # 1. Dataset root/annotation
        self.root = annotation_root
        self.data = self.__load_narration()
        self.PASSES = [1, 2]

        # 2. let's load ego4d metadata
        self.meta_data = Metadata(annotation_root)

    
    def __load_narration(self):
        # return vid -> goalstep_data
        narration = load_json(os.path.join(self.root, "narration.json"))
        return narration

    def __is_grouped_video(self, vid):
        return self.meta_data.is_grouped(vid)

    def __handle_grp(self, grp_vid, shift_unmapped=False, shift_frames=True):
        """
        Build synthetic narrations for a grouped (grp-*) video by concatenating narrations of
        its underlying group_video_uids and shifting timestamps by cumulative video durations.

        Uses the grouped video's FPS as the reference FPS for timestamp_frame shifting.
        """
        group_vids = self.meta_data.get_group_vids(grp_vid)  # [A, B, ...] in concat order

        # Reference FPS: use the grouped video's FPS (matches the actual grp mp4 timeline)
        try:
            fps_ref = float(self.meta_data.get_fps(grp_vid))
        except Exception:
            fps_ref = 30.0  # safe default; can also set shift_frames=False if you prefer

        # Deep-copy to avoid mutating raw self.data entries (important!)
        parts = [copy.deepcopy(self.data[vid]) for vid in group_vids]

        # Offsets: [0, dur(A), dur(A)+dur(B), ...]
        durations = [float(self.meta_data.get_video_duration(vid)) for vid in group_vids]
        offsets = [0.0]
        for d in durations[:-1]:
            offsets.append(offsets[-1] + d)

        # Shift times inside each part
        for part, off, src_uid in zip(parts, offsets, group_vids):
            frame_off = int(round(off * fps_ref)) if shift_frames else 0

            for pass_nr in self.PASSES:
                p = part.get(f"narration_pass_{pass_nr}", {})
                narrs = p.get("narrations", [])
                sums  = p.get("summaries", [])

                for line in narrs:
                    # seconds
                    if line.get("timestamp_sec") is not None:
                        line["timestamp_sec"] = float(line["timestamp_sec"]) + off

                    # optional: keep provenance OR shift this too (your choice)
                    if shift_unmapped and line.get("_unmapped_timestamp_sec") is not None:
                        line["_unmapped_timestamp_sec"] = float(line["_unmapped_timestamp_sec"]) + off

                    # frames (in reference FPS of grp video)
                    if shift_frames and line.get("timestamp_frame") is not None:
                        line["timestamp_frame"] = int(line["timestamp_frame"]) + frame_off

                    # provenance (super helpful for debugging)
                    line["source_video_uid"] = src_uid
                    line["source_offset_sec"] = off

                for s in sums:
                    if s.get("start_sec") is not None:
                        s["start_sec"] = float(s["start_sec"]) + off
                    if s.get("end_sec") is not None:
                        s["end_sec"] = float(s["end_sec"]) + off

                    s["source_video_uid"] = src_uid
                    s["source_offset_sec"] = off

        # Merge
        result = {f"narration_pass_{x}": {"narrations": [], "summaries": []} for x in self.PASSES}
        for part in parts:
            for pass_nr in self.PASSES:
                p = part.get(f"narration_pass_{pass_nr}", {})
                result[f"narration_pass_{pass_nr}"]["narrations"].extend(p.get("narrations", []))
                result[f"narration_pass_{pass_nr}"]["summaries"].extend(p.get("summaries", []))

        # Sort
        for pass_nr in self.PASSES:
            result[f"narration_pass_{pass_nr}"]["narrations"].sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))
            result[f"narration_pass_{pass_nr}"]["summaries"].sort(key=lambda x: float(x.get("start_sec", 0.0)))

        result["status"] = "synthetic_grouped"
        result["fps_ref"] = fps_ref  # optional debug info
        return result

    
    def get_string(self, vid, pass_nr=1, timeformat="seconds"):
        # 1) Resolve data
        if self.__is_grouped_video(vid):
            data = self.__handle_grp(grp_vid=vid)
        elif not self.data.get(vid):
            logging.error(f"Could not find {vid} in narrations. Silently return empty string.")
            return ""
        else:
            data = self.data[vid]

        # 2) Validate pass exists
        pass_key = f"narration_pass_{pass_nr}"
        if not data.get(pass_key):
            logging.error(
                f"Could not find {pass_key} in video with id={vid}. "
                f"Accessible keys={list(data.keys())}."
            )
            return ""

        narrs = data[pass_key].get("narrations", [])
        if not narrs:
            return ""

        # 3) Sort by time (do not mutate underlying list)
        narrs_sorted = sorted(narrs, key=lambda n: float(n.get("timestamp_sec", 0.0)))

        # 4) Format all lines
        return "\n".join(format_narration_line(n, timeformat) for n in narrs_sorted)

    
    def get_string_in_range(self, vid, left, right, pass_nr=1, timeformat="seconds"):
        # 1) Resolve data
        if self.__is_grouped_video(vid):
            data = self.__handle_grp(grp_vid=vid)
        elif not self.data.get(vid):
            logging.error(f"Could not find {vid} in narrations. Silently return empty string.")
            return ""
        else:
            data = self.data[vid]

        # 2) Validate pass exists (fix log: for grp vids, self.data[vid] may not exist)
        pass_key = f"narration_pass_{pass_nr}"
        if not data.get(pass_key):
            logging.error(
                f"Could not find {pass_key} in video with id={vid}. "
                f"Accessible keys={list(data.keys())}."
            )
            return ""

        # 3) Get narrations and sort by time (do NOT mutate original)
        narrs = data[pass_key].get("narrations", [])
        if not narrs:
            return ""

        narrs_sorted = sorted(narrs, key=lambda n: float(n.get("timestamp_sec", 0.0)))

        # 4) Collect lines in range (early-break enabled because list is sorted)
        result = []
        for n in narrs_sorted:
            t = float(n["timestamp_sec"])
            if t < left:
                continue
            if t > right:
                break
            result.append(format_narration_line(n, timeformat))

        return "\n".join(result)
    
    def get_string_cut_out(self, vid, left, right, pass_nr=1, timeformat="seconds"):
        # 1) Resolve data
        if self.__is_grouped_video(vid):
            data = self.__handle_grp(grp_vid=vid)
        elif not self.data.get(vid):
            logging.error(f"Could not find {vid} in narrations. Silently return empty string.")
            return ""
        else:
            data = self.data[vid]

        # 2) Validate pass exists (fix log: for grp vids, self.data[vid] may not exist)
        pass_key = f"narration_pass_{pass_nr}"
        if not data.get(pass_key):
            logging.error(
                f"Could not find {pass_key} in video with id={vid}. "
                f"Accessible keys={list(data.keys())}."
            )
            return ""

        # 3) Get narrations and sort by time (do NOT mutate original)
        narrs = data[pass_key].get("narrations", [])
        if not narrs:
            return ""

        narrs_sorted = sorted(narrs, key=lambda n: float(n.get("timestamp_sec", 0.0)))

        # 4) Collect lines in range (early-break enabled because list is sorted)
        result = []
        for n in narrs_sorted:
            t = float(n["timestamp_sec"])
            # idea: only skip lines in dedicated range
            if left < t < right:
                continue
            result.append(format_narration_line(n, timeformat))

        return "\n".join(result)
    
    def get_string_in_range_w_cut_out(self, vid, rleft, rright, cleft, cright, pass_nr=1, timeformat="seconds"):
        # 1) Resolve data
        if self.__is_grouped_video(vid):
            data = self.__handle_grp(grp_vid=vid)
        elif not self.data.get(vid):
            logging.error(f"Could not find {vid} in narrations. Silently return empty string.")
            return ""
        else:
            data = self.data[vid]

        # 2) Validate pass exists (fix log: for grp vids, self.data[vid] may not exist)
        pass_key = f"narration_pass_{pass_nr}"
        if not data.get(pass_key):
            logging.error(
                f"Could not find {pass_key} in video with id={vid}. "
                f"Accessible keys={list(data.keys())}."
            )
            return ""

        # 3) Get narrations and sort by time (do NOT mutate original)
        narrs = data[pass_key].get("narrations", [])
        if not narrs:
            return ""

        narrs_sorted = sorted(narrs, key=lambda n: float(n.get("timestamp_sec", 0.0)))

        # 4) Collect lines in range (early-break enabled because list is sorted)
        result = []
        for n in narrs_sorted:
            t = float(n["timestamp_sec"])
            if t < rleft:
                continue
            if cleft <= t <= cright:
                continue
            if t > rright:
                break
            result.append(format_narration_line(n, timeformat))

        return "\n".join(result)
    
    def get_strings_in_interval(self, vid, interval, overlap=0, pass_nr=1, timeformat="seconds", left=0.0, right=None, drop_empty=True):
        """
        Yields a list[str] per window (formatted narration lines).
        Windows are [start, end) to avoid duplicates at boundaries.
        """
        if interval <= 0:
            raise ValueError("interval must be > 0")
        if overlap < 0 or overlap >= interval:
            raise ValueError("overlap must be >= 0 and < interval")

        stride = interval - overlap

        # If right not provided, use last narration timestamp (or 0.0)
        if right is None:
            # resolve data the same way as your other methods
            if self.__is_grouped_video(vid):
                data = self.__handle_grp(grp_vid=vid)
            else:
                data = self.data.get(vid)
            if not data:
                logging.error(f"Could not find {vid} in narrations. Silently stop.")
                return

            pass_key = f"narration_pass_{pass_nr}"
            narrs = data.get(pass_key, {}).get("narrations", [])
            if not narrs:
                return
            right = max(float(n.get("timestamp_sec", 0.0)) for n in narrs)

        t = float(left)
        right = float(right)

        while t < right:
            end = min(t + float(interval), right)

            # your get_string_in_range is inclusive on 'right', so make end exclusive
            end_adj = end - 1e-6 if end < right else end

            s = self.get_string_in_range(
                vid=vid,
                left=t,
                right=end_adj,
                pass_nr=pass_nr,
                timeformat=timeformat,
            )

            if (not drop_empty) or (s and s.strip()):
                yield s

            #lines = s.splitlines() if s else []
            #if (not drop_empty) or lines:
            #    yield s

            t += float(stride)

        

    


if __name__ == "__main__":
    gs = Narrations("/nfs/data8/liao/meikel/datasets/ego4d/v2/annotations")

    #print(gs.get_string_cutoff("5dde766e-e93a-4d48-b3c7-96774f24d39d",cutoff=600))
    #print(gs.get_string_in_range("5dde766e-e93a-4d48-b3c7-96774f24d39d",left=530, right=600, pass_nr=3))
    #vid = "grp-deb43adb-2e2f-454a-ba4e-811ebfe4eaa2"
    vid = "d730a47a-013d-4cd3-b478-370f2fb0d918"
    #print(gs.get_string_in_range(vid, left=2*60, right=32*60, pass_nr=1))
    #print(gs.get_string(vid))
    #print("--"*50)
    #print(gs.get_string_cut_out(vid, left=2*60+30, right=14*60, pass_nr=1, timeformat="hhmmss"))
    #print("--"*50)
    #print(gs.get_string_in_range_w_cut_out(vid, rleft=2*60+30, rright=14*60, cleft=10*60, cright=12*60+2, pass_nr=1, timeformat="hhmmss"))
    for n in gs.get_strings_in_interval(vid, interval=3*60, overlap=30, timeformat="seconds"):
        print(n)
        input()