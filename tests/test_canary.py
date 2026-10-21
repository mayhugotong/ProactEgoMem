from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.support import install_common_stubs


install_common_stubs()

from proact.data.canary import CANARY_KEY, CANARY_STRING, has_canary, main, stamp_file, stamp_text


RELEASE = {
    "schema_version": "v3",
    "task": "state_tracking",
    "videos": {"vid-a": {"samples": {"s1": {"times": {"start_sec": 1.0, "end_sec": 2.0}}}}},
    "clips": {"clip-a": {"video_uid": "vid-a", "sample_ids": ["s1"]}},
}


class StampTextTests(unittest.TestCase):
    def test_indented_file_keeps_every_other_byte(self):
        text = json.dumps(RELEASE, indent=2) + "\n"
        out = stamp_text(text)

        self.assertEqual(list(json.loads(out))[0], CANARY_KEY)
        self.assertEqual(json.loads(out), {CANARY_KEY: CANARY_STRING, **RELEASE})
        canary_line = f'\n  "{CANARY_KEY}": {json.dumps(CANARY_STRING)},'
        self.assertEqual(out.replace(canary_line, "", 1), text)

    def test_compact_file(self):
        out = stamp_text(json.dumps(RELEASE))
        self.assertTrue(has_canary(json.loads(out)))
        self.assertNotIn("\n", out)

    def test_stamping_twice_is_a_no_op(self):
        once = stamp_text(json.dumps(RELEASE, indent=4))
        self.assertEqual(stamp_text(once), once)

    def test_refuses_legacy_grouped_format(self):
        grouped = {"vid-a": {"matches": []}}
        with self.assertRaises(ValueError):
            stamp_text(json.dumps(grouped))

    def test_refuses_a_different_canary(self):
        with self.assertRaises(ValueError):
            stamp_text(json.dumps({CANARY_KEY: "other", **RELEASE}))


class CommandLineTests(unittest.TestCase):
    def test_stamp_then_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_tracking.json"
            path.write_text(json.dumps(RELEASE, indent=2), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--check", str(path)]), 1)
                self.assertTrue(stamp_file(path))
                self.assertFalse(stamp_file(path))
                self.assertEqual(main(["--check", str(path)]), 0)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["state_tracking.json"])


if __name__ == "__main__":
    unittest.main()
