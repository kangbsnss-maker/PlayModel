"""Negative controls for evidence required before source image cleanup."""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_preparation.py"
SPEC = importlib.util.spec_from_file_location("verify_preparation", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        research = self.root / "docs/research"
        research.mkdir(parents=True)
        self.notes = research / "page-notes.md"
        self.notes.write_text("\n".join(f"## page-{n:02d}\n" + "Evidence statement. " * 20 for n in range(1, 30)), encoding="utf-8")
        self.raw = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1, 1) + b"fixture remainder"
        self.manifest = {
            "schema_version": 1, "source_count": 29,
            "pages": [
                {"page": n, "filename": f"source-{n}.png", "sha256": hashlib.sha256(self.raw).hexdigest(),
                 "width": 1, "height": 1, "bytes": len(self.raw), "title": "Test source",
                 "notes": f"docs/research/page-notes.md#page-{n:02d}"}
                for n in range(1, 30)
            ],
        }
        self.manifest_path = research / "source-manifest.json"
        self.write_manifest()
        for page in self.manifest["pages"]:
            (self.root / page["filename"]).write_bytes(self.raw)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def write_receipt(self):
        receipt = {
            "status": "completed", "pre_delete_verification": "PREPARATION_VERIFIED",
            "deleted_sources": [{"filename": p["filename"], "sha256": p["sha256"]} for p in self.manifest["pages"]],
        }
        (self.root / "docs/source-cleanup.json").write_text(json.dumps(receipt), encoding="utf-8")

    def remove_sources(self):
        for page in self.manifest["pages"]:
            (self.root / page["filename"]).unlink()

    def test_original_hash_mismatch_fails_even_when_notes_exist(self):
        source = self.root / "source-1.png"
        source.write_bytes(self.raw)
        verify.verify_sources(self.root)
        source.write_bytes(b"X" * len(self.raw))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify.verify_sources(self.root)

    def test_missing_page_notes_fail_even_after_originals_removed(self):
        self.remove_sources()
        self.write_receipt()
        verify.verify_sources(self.root)
        self.notes.write_text(self.notes.read_text().replace("## page-15\n", "## omitted\n"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing durable note"):
            verify.verify_sources(self.root)

    def test_duplicate_page_is_not_complete_coverage(self):
        self.manifest["pages"][-1]["page"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            verify.verify_sources(self.root)

    def test_deleted_images_without_receipt_do_not_prove_cleanup(self):
        self.remove_sources()
        with self.assertRaises(FileNotFoundError):
            verify.verify_cleanup(self.root, self.manifest)
        self.write_receipt()
        verify.verify_cleanup(self.root, self.manifest)

    def test_surviving_original_prevents_completed_cleanup(self):
        self.remove_sources()
        self.write_receipt()
        verify.verify_cleanup(self.root, self.manifest)
        (self.root / "source-1.png").write_bytes(self.raw)
        with self.assertRaisesRegex(ValueError, "still present"):
            verify.verify_cleanup(self.root, self.manifest)

    def test_missing_original_before_cleanup_is_failure(self):
        verify.verify_sources(self.root)
        (self.root / "source-8.png").unlink()
        with self.assertRaisesRegex(ValueError, "Source missing before"):
            verify.verify_sources(self.root)

    def test_manifest_dimensions_must_match_original(self):
        self.manifest["pages"][0]["width"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            verify.verify_sources(self.root)


if __name__ == "__main__":
    unittest.main()
