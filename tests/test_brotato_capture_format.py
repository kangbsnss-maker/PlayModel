import hashlib
from pathlib import Path
import tempfile
import unittest

from playmodel.games.brotato.capture import _png, _sample_bgra, read_diagnostic_png, region_digest


class CaptureFormatTests(unittest.TestCase):
    def test_raw_sampling_matches_original_positions_and_stride_one_reuses_bytes(self):
        for width, height in ((7, 5), (320, 180), (1, 1)):
            raw = bytes((index * 31) % 256 for index in range(width * height * 4))
            for stride in (1, 2, 6, 32):
                with self.subTest(size=(width, height), stride=stride):
                    pixels, sw, sh = _sample_bgra(raw, width, height, stride)
                    expected = b''.join(raw[(y * width + x) * 4:(y * width + x + 1) * 4]
                                        for y in range(0, height, stride) for x in range(0, width, stride))
                    self.assertEqual(pixels, expected)
                    self.assertEqual((sw, sh), ((width + stride - 1) // stride, (height + stride - 1) // stride))
                    if stride == 1:
                        self.assertIs(pixels, raw)

    def test_saved_frame_and_odd_stride_preserve_rgb_positions(self):
        width, height = 7, 5
        source = bytes(channel for y in range(height) for x in range(width)
                       for channel in (x * 10, y * 20, x + y, 0))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(_png(width, height, source))
            self.assertEqual(read_diagnostic_png(image), (source, width, height))
            sampled, sw, sh = read_diagnostic_png(image, stride=3)
            expected = b"".join(source[(y * width + x) * 4:(y * width + x + 1) * 4]
                                for y in (0, 3) for x in (0, 3, 6))
            self.assertEqual((sampled, sw, sh), (expected, 3, 2))
            rgb = b"".join(bytes((x + y, y * 20, x * 10)) for y in (1, 2) for x in (2, 3, 4))
            self.assertEqual(region_digest(image, (2, 1, 5, 3)), hashlib.sha256(rgb).hexdigest())

    def test_invalid_stride_fails_before_file_access(self):
        for value in (0, 33, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_diagnostic_png(Path("missing.png"), stride=value)


if __name__ == "__main__":
    unittest.main()
