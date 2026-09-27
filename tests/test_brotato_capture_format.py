import hashlib
from pathlib import Path
import tempfile
import unittest

from playmodel.games.brotato.capture import _png, read_diagnostic_png, region_digest


class CaptureFormatTests(unittest.TestCase):
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
