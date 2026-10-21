from proact.utils.io import load_json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metadata:
    def __init__(self, annotation_root):
        # 1. Dataset root/annotation
        self.root = annotation_root

        # 2. let's load ego4d metadata
        self.ego4d_meta_data = load_json(Path(self.root).resolve().parents[1] / "ego4d.json")

        # 3. make meta data accessible by video_id
        self.videos_meta_data = dict()
        self.group_member_to_group_uid = dict()

        self.init_vid_indexing()  # this fills the videos_meta_data dict
        self.init_group_member_indexing()
    
    def init_vid_indexing(self):
        for video in self.ego4d_meta_data["videos"]:
            vid = video.get("video_uid")
            if vid in self.videos_meta_data:
                logger.warning("VID collision detected. This means an entry will be discarded in video to metadata entry. You might want to losen constraints. NOTE: this is a developer note. User is doing nothing wrong. Open github issue pls. Or even better pull request thank you~")
            
            self.videos_meta_data[video.pop("video_uid")] = video

    def init_group_member_indexing(self):
        for group_video_uid, metadata in self.videos_meta_data.items():
            if not metadata.get("is_grouped"):
                continue

            for member_uid in metadata.get("group_video_uids") or []:
                existing_parent = self.group_member_to_group_uid.get(member_uid)
                if existing_parent is not None and existing_parent != group_video_uid:
                    logger.warning(
                        "Grouped video member collision detected for %s. "
                        "Keeping existing parent %s and ignoring %s.",
                        member_uid,
                        existing_parent,
                        group_video_uid,
                    )
                    continue
                self.group_member_to_group_uid[member_uid] = group_video_uid
    
    def get(self, vid):
        return self.videos_meta_data[vid]
    
    ### get video duration
    def get_video_duration(self, vid):
        return self.get(vid)["video_metadata"]["video_duration_sec"]
    
    ### This is for handling grouped videos
    def is_grouped(self, vid):
        metadata = self.get(vid)

        return metadata["is_grouped"]
    
    def get_group_vids(self, vid):
        # returns NONE if .is_grouped is false!!
        return self.get(vid).get("group_video_uids")

    def get_parent_group_video_uid(self, vid):
        return self.group_member_to_group_uid.get(vid)

    def resolve_grouped_video_uid(self, vid):
        if self.is_grouped(vid):
            return vid
        return self.get_parent_group_video_uid(vid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Ego4D metadata. Pass a video UID to resolve grouped-video "
            "membership, or run without one to print the legacy subset stats."
        )
    )
    parser.add_argument(
        "video_uid",
        nargs="?",
        help="Optional video UID to inspect for grouped-video membership.",
    )
    parser.add_argument(
        "--annotation-root",
        default="/nfs/data8/liao/meikel/datasets/ego4d/v2/annotations",
        help="Path to the Ego4D annotation root used by Metadata.",
    )
    args = parser.parse_args()

    m = Metadata(args.annotation_root)

    if args.video_uid is not None:
        if args.video_uid not in m.videos_meta_data:
            parser.error(f"Video UID not found in Ego4D metadata: {args.video_uid}")

        is_grouped = m.is_grouped(args.video_uid)
        parent_group_uid = m.get_parent_group_video_uid(args.video_uid)
        resolved_group_uid = m.resolve_grouped_video_uid(args.video_uid)

        print(f"video_uid: {args.video_uid}")
        print(f"is_grouped_video_uid: {is_grouped}")
        print(f"member_of_grouped_video_uid: {parent_group_uid}")
        print(f"resolved_grouped_video_uid: {resolved_group_uid}")

        if is_grouped:
            print(f"group_member_video_uids: {m.get_group_vids(args.video_uid) or []}")

        raise SystemExit(0)

    #print(gs.get_string_cutoff("5dde766e-e93a-4d48-b3c7-96774f24d39d",cutoff=600))
    #print(gs.get_string_in_range("5dde766e-e93a-4d48-b3c7-96774f24d39d",left=530, right=600, pass_nr=3))
    vid1 = "grp-3549e738-4cce-492f-a530-e473777e3220"
    vid2 = "db6c7c04-6678-4748-9e66-b3a43b0dd37d"

    print(m.is_grouped(vid1))
    print(m.get_group_vids(vid1))

    print("+"*50)

    print(m.is_grouped(vid2))
    print(m.get_group_vids(vid2))

    print("+"*50)

    print(m.get(vid2))


    from collections import Counter
    from statistics import mean, median

    JSON_FILES = [
        "/path/to/proact-benchmark/data/after_trim_25_03_26/object_tracking.json",
        "/path/to/proact-benchmark/data/after_trim_25_03_26/state_tracking.json",
        "/path/to/proact-benchmark/data/after_trim_25_03_26/video_capture.json"
    ]

    TOP_K = 25  # how many top entries to print for scenarios / objects / etc.

    def pct(n, d):
        return 100.0 * n / d if d else 0.0

    def fmt_pct(n, d):
        return f"{pct(n, d):6.2f}%"

    def fmt_seconds(sec):
        if sec is None:
            return "n/a"
        sec = float(sec)
        sign = "-" if sec < 0 else ""
        sec = abs(sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{sign}{h:02d}:{m:02d}:{s:06.3f}"

    def print_counter(title, counter, denom=None, top_k=None, note=None):
        print("\n" + title)
        print("-" * len(title))
        if not counter:
            print("  <empty>")
            return

        items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
        if top_k is not None:
            items = items[:top_k]

        key_width = max(len(str(k)) for k, _ in items)
        for key, count in items:
            if denom is None:
                print(f"  {str(key):<{key_width}} : {count}")
            else:
                print(f"  {str(key):<{key_width}} : {count:>6}  ({fmt_pct(count, denom)})")

        if note:
            print(f"  NOTE: {note}")

    def print_duration_stats(title, values):
        print("\n" + title)
        print("-" * len(title))
        if not values:
            print("  <empty>")
            return

        values = [float(v) for v in values]
        print(f"  n      : {len(values)}")
        print(f"  total  : {fmt_seconds(sum(values))}")
        print(f"  mean   : {fmt_seconds(mean(values))}")
        print(f"  median : {fmt_seconds(median(values))}")
        print(f"  min    : {fmt_seconds(min(values))}")
        print(f"  max    : {fmt_seconds(max(values))}")

    def load_subset_jsons(json_files):
        """
        Merges multiple JSON files of the form:
        {
          "<video_uid>": {
            "matches": [...]
          },
          ...
        }
        """
        merged = {}

        for json_file in json_files:
            json_file = str(json_file)
            payload = load_json(json_file)

            if not isinstance(payload, dict):
                logger.warning(f"Skipping non-dict payload in {json_file}")
                continue

            for top_video_uid, entry in payload.items():
                slot = merged.setdefault(
                    top_video_uid,
                    {
                        "matches": [],
                        "json_files": set(),
                    },
                )
                slot["json_files"].add(json_file)

                if not isinstance(entry, dict):
                    logger.warning(f"Entry for {top_video_uid} in {json_file} is not a dict; skipping.")
                    continue

                matches = entry.get("matches", []) or []
                if not isinstance(matches, list):
                    logger.warning(f"'matches' for {top_video_uid} in {json_file} is not a list; skipping.")
                    matches = []

                slot["matches"].extend(matches)

        return merged

    def print_subset_publication_stats(metadata_obj, json_files, top_k=25):
        subset = load_subset_jsons(json_files)

        video_uids = sorted(subset.keys())
        n_unique_videos = len(video_uids)

        if n_unique_videos == 0:
            print("\nNo videos found. Did you fill JSON_FILES?")
            return

        # ----------------------------
        # subset-level / match-level stats
        # ----------------------------
        n_json_files = len(json_files)
        n_videos_in_multiple_files = sum(1 for x in subset.values() if len(x["json_files"]) > 1)

        total_match_rows = 0
        total_gt_match_rows = 0
        total_assigned_to_gt_rows = 0
        all_object_counter = Counter()
        gt_object_counter = Counter()
        unique_clip_ids = set()
        clip_durations = []
        topkey_inner_vid_mismatch = 0

        for top_video_uid, entry in subset.items():
            for match in entry["matches"]:
                total_match_rows += 1

                if match.get("is_gt", False):
                    total_gt_match_rows += 1

                if match.get("is_assigned_to_gt", False):
                    total_assigned_to_gt_rows += 1

                obj = match.get("object") or "<missing>"
                all_object_counter[obj] += 1
                if match.get("is_gt", False):
                    gt_object_counter[obj] += 1

                s = match.get("start_time_sec")
                e = match.get("end_time_sec")
                if s is not None and e is not None:
                    try:
                        clip_durations.append(max(0.0, float(e) - float(s)))
                    except Exception:
                        pass

                clip_info = match.get("clip_information") or {}
                clip_uid = clip_info.get("clip_uid")
                if clip_uid:
                    unique_clip_ids.add(clip_uid)
                else:
                    unique_clip_ids.add(
                        (
                            top_video_uid,
                            match.get("start_time_sec"),
                            match.get("end_time_sec"),
                        )
                    )

                inner_video_uid = match.get("video_uid")
                if inner_video_uid is not None and inner_video_uid != top_video_uid:
                    topkey_inner_vid_mismatch += 1

        # ----------------------------
        # metadata-driven video stats
        # ----------------------------
        missing_metadata = []

        scenario_counter = Counter()
        source_counter = Counter()
        device_counter = Counter()
        physical_setting_counter = Counter()

        grouped_counter = Counter()
        redacted_counter = Counter()
        stereo_counter = Counter()
        imu_counter = Counter()
        gaze_counter = Counter()

        split_counters = {
            "split_em": Counter(),
            "split_av": Counter(),
            "split_fho": Counter(),
            "split_goalstep": Counter(),
        }

        video_durations = []

        for vid in video_uids:
            if vid not in metadata_obj.videos_meta_data:
                missing_metadata.append(vid)
                continue

            md = metadata_obj.get(vid)

            scenarios = md.get("scenarios") or []
            if scenarios:
                # use set(...) so a weird duplicated scenario string in one video
                # does not count twice for the same video
                scenario_counter.update(set(scenarios))
            else:
                scenario_counter["<missing>"] += 1

            source_counter[md.get("video_source") or "<missing>"] += 1
            device_counter[md.get("device") or "<missing>"] += 1

            if md.get("physical_setting_name") is not None:
                physical_setting_counter[md.get("physical_setting_name")] += 1

            grouped_counter[str(md.get("is_grouped"))] += 1
            redacted_counter[str(md.get("has_redacted_regions"))] += 1
            stereo_counter[str(md.get("is_stereo"))] += 1
            imu_counter[str(md.get("has_imu"))] += 1
            gaze_counter[str(md.get("has_gaze"))] += 1

            for split_name, split_counter in split_counters.items():
                split_val = md.get(split_name)
                if split_val is not None:
                    split_counter[str(split_val)] += 1

            dur = None
            if isinstance(md.get("video_metadata"), dict):
                dur = md["video_metadata"].get("video_duration_sec")
            if dur is None:
                dur = md.get("duration_sec")
            if dur is not None:
                try:
                    video_durations.append(float(dur))
                except Exception:
                    pass

        n_found_metadata = n_unique_videos - len(missing_metadata)

        # ----------------------------
        # printing
        # ----------------------------
        print("\n" + "=" * 90)
        print("SUBSET / PUBLICATION STATS")
        print("=" * 90)

        print("\nOverview")
        print("--------")
        print(f"  input JSON files               : {n_json_files}")
        print(f"  unique videos                  : {n_unique_videos}")
        print(f"  videos with metadata           : {n_found_metadata}")
        print(f"  videos missing metadata        : {len(missing_metadata)}")
        print(f"  videos in >1 input file        : {n_videos_in_multiple_files}")
        print(f"  unique clips                   : {len(unique_clip_ids)}")
        print(f"  total match rows               : {total_match_rows}")
        print(f"  GT match rows                  : {total_gt_match_rows} ({fmt_pct(total_gt_match_rows, total_match_rows)})")
        print(f"  assigned-to-GT rows            : {total_assigned_to_gt_rows} ({fmt_pct(total_assigned_to_gt_rows, total_match_rows)})")
        print(f"  topkey/inner video_uid mismatch: {topkey_inner_vid_mismatch}")

        if missing_metadata:
            print("\nMissing metadata video_uids (first 20)")
            print("--------------------------------------")
            for vid in missing_metadata[:20]:
                print(f"  {vid}")
            if len(missing_metadata) > 20:
                print(f"  ... and {len(missing_metadata) - 20} more")

        print_duration_stats("Full-video duration stats (over unique videos)", video_durations)
        print_duration_stats("Matched-segment duration stats (over match rows)", clip_durations)

        print_counter(
            "Scenarios (share of unique videos)",
            scenario_counter,
            denom=n_unique_videos,
            top_k=top_k,
            note="Multi-label field: percentages can sum to >100% if videos have multiple scenarios.",
        )

        print_counter(
            "Video sources (share of unique videos)",
            source_counter,
            denom=n_unique_videos,
            top_k=top_k,
        )

        print_counter(
            "Devices (share of unique videos)",
            device_counter,
            denom=n_unique_videos,
            top_k=top_k,
        )

        if physical_setting_counter:
            print_counter(
                "Physical settings (share of unique videos)",
                physical_setting_counter,
                denom=n_unique_videos,
                top_k=top_k,
            )

        print_counter("Grouped videos", grouped_counter, denom=n_unique_videos)
        print_counter("Has redacted regions", redacted_counter, denom=n_unique_videos)
        print_counter("Stereo videos", stereo_counter, denom=n_unique_videos)
        print_counter("Has IMU", imu_counter, denom=n_unique_videos)
        print_counter("Has gaze", gaze_counter, denom=n_unique_videos)

        for split_name, split_counter in split_counters.items():
            if split_counter:
                denom = sum(split_counter.values())
                print_counter(
                    f"{split_name} (share among videos where this split is present)",
                    split_counter,
                    denom=denom,
                    top_k=top_k,
                )

        print_counter(
            "Top matched objects (share of match rows)",
            all_object_counter,
            denom=total_match_rows,
            top_k=top_k,
        )

        if total_gt_match_rows > 0:
            print_counter(
                "Top GT objects (share of GT match rows)",
                gt_object_counter,
                denom=total_gt_match_rows,
                top_k=top_k,
            )

        print("\nDone.")
        print("=" * 90)

    # ---- run it ----
    print_subset_publication_stats(m, JSON_FILES, top_k=TOP_K)
