from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest import mock

from tests.support import install_common_stubs

install_common_stubs()

if importlib.util.find_spec("cv2") is None:
    cv2 = types.ModuleType("cv2")
    cv2.INTER_AREA = 1
    cv2.INTER_LINEAR = 2
    cv2.resize = lambda frame, size, interpolation=None: frame
    sys.modules["cv2"] = cv2

if importlib.util.find_spec("av") is None:
    av = types.ModuleType("av")
    av.time_base = 1_000_000
    av.AVError = Exception
    av.open = lambda path: None
    sys.modules["av"] = av

from proact.utils import media


class _Stream:
    type = "video"
    time_base = 1.0
    duration = 2.0
    width = 1
    height = 1


class _Container:
    streams = [_Stream()]
    duration = None

    def close(self):
        return None


class MediaSamplingBackendTests(unittest.TestCase):
    """Backend selection is explicit: ffmpeg-cuda (default) fails loudly,
    pyav-cpu must be requested deliberately, and there is NO auto mode and
    NO silent fallback in either direction."""

    def _open(self, path):
        return _Container()

    def test_cuda_backend_reports_selected_method(self):
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "_ffmpeg_cuda_hwaccel_available", return_value=True),
            mock.patch.object(media, "_ffmpeg_filter_available", return_value=True),
            mock.patch.object(media, "_sample_frames_at_fps_ffmpeg_gpu", return_value=["gpu-frame"]),
        ):
            frames, metadata = media.sample_frames_at_fps_with_metadata(
                "clip.mp4", 1.0, backend="ffmpeg-cuda",
                resize_width=336, resize_height=336,
            )

        self.assertEqual(frames, ["gpu-frame"])
        self.assertEqual(metadata["selected_backend"], "ffmpeg-cuda")
        self.assertEqual(metadata["processing_path"], "cuda-decode+cuda-scale+cpu-fps")
        self.assertTrue(metadata["gpu_resize"])

    def test_auto_backend_is_rejected(self):
        with mock.patch.object(media.av, "open", side_effect=self._open):
            with self.assertRaisesRegex(ValueError, "Unsupported frame sampling backend"):
                media.sample_frames_at_fps_with_metadata("clip.mp4", 1.0, backend="auto")

    def test_cuda_decode_error_propagates_without_cpu_fallback(self):
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "_resolve_ffmpeg_path", return_value="/ffmpeg"),
            mock.patch.object(media, "_ffmpeg_cuda_hwaccel_available", return_value=True),
            mock.patch.object(media, "_ffmpeg_filter_available", return_value=False),
            mock.patch.object(
                media,
                "_sample_frames_at_fps_ffmpeg_gpu",
                side_effect=RuntimeError("decoder failed"),
            ),
            mock.patch.object(media, "sample_uniform_frames_memory_friendly") as cpu_sampler,
        ):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                media.sample_frames_at_fps_with_metadata(
                    "clip.mp4", 1.0, backend="ffmpeg-cuda",
                )

        cpu_sampler.assert_not_called()

    def test_explicit_cpu_backend_uses_pyav(self):
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "sample_uniform_frames_memory_friendly", return_value=["cpu-frame"]),
        ):
            frames, metadata = media.sample_frames_at_fps_with_metadata(
                "clip.mp4", 1.0, backend="pyav-cpu",
            )

        self.assertEqual(frames, ["cpu-frame"])
        self.assertEqual(metadata["selected_backend"], "pyav-cpu")

    def test_cuda_accepts_fewer_frames_on_target_grid(self):
        # ffmpeg returned 2 frames for 3 grid targets (end-exclusive fps filter):
        # the decoded frames are kept and aligned to the leading grid targets.
        ffmpeg_result = mock.Mock(
            returncode=0,
            stdout=bytes(range(6)),
            stderr=b"",
        )
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "_resolve_ffmpeg_path", return_value="/ffmpeg"),
            mock.patch.object(media, "_ffmpeg_filter_available", return_value=False),
            mock.patch.object(media.subprocess, "run", return_value=ffmpeg_result),
            self.assertLogs(media.logger, level="INFO") as captured,
        ):
            frames = media._sample_frames_at_fps_ffmpeg_gpu(
                path="clip.mp4",
                fps=1.0,
                S0_sec=0.0,
                S1_sec=2.5,
                include_endpoints=True,
                convert_to_rgb=True,
                resize_width=1,
                resize_height=1,
            )

        self.assertEqual(len(frames), 2)
        self.assertEqual([frame.time for frame in frames], [0.0, 1.25])
        self.assertTrue(all(frame.array.flags.owndata for frame in frames))
        self.assertIn("accepting the decoded frames", "\n".join(captured.output))

    def test_cuda_accepts_more_frames_by_truncating_to_targets(self):
        # ffmpeg returned 3 frames for 2 grid targets: keep the first 2, never raise.
        ffmpeg_result = mock.Mock(
            returncode=0,
            stdout=bytes(range(9)),
            stderr=b"",
        )
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "_resolve_ffmpeg_path", return_value="/ffmpeg"),
            mock.patch.object(media, "_ffmpeg_filter_available", return_value=False),
            mock.patch.object(media.subprocess, "run", return_value=ffmpeg_result),
            self.assertLogs(media.logger, level="INFO") as captured,
        ):
            frames = media._sample_frames_at_fps_ffmpeg_gpu(
                path="clip.mp4",
                fps=1.0,
                S0_sec=0.0,
                S1_sec=1.0,
                include_endpoints=True,
                convert_to_rgb=True,
                resize_width=1,
                resize_height=1,
            )

        self.assertEqual(len(frames), 2)
        self.assertEqual([frame.time for frame in frames], [0.0, 1.0])
        self.assertIn("keeping the first", "\n".join(captured.output))

    def test_required_cuda_fails_without_silent_cpu_fallback(self):
        with (
            mock.patch.object(media.av, "open", side_effect=self._open),
            mock.patch.object(media, "_ffmpeg_cuda_hwaccel_available", return_value=False),
            mock.patch.object(media, "sample_uniform_frames_memory_friendly") as cpu_sampler,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not advertise"):
                media.sample_frames_at_fps_with_metadata(
                    "clip.mp4", 1.0, backend="ffmpeg-cuda",
                )

        cpu_sampler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
