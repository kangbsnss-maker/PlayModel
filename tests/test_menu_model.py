"""Learning, evidence and abstention contracts; synthetic pixels are not game QA."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from playmodel.games.brotato.capture import _png, read_diagnostic_png
from playmodel.games.brotato.menu_model import MenuClassifier, train_manifest


class MenuModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        samples = []
        for index, split in enumerate(('train', 'validation', 'test')):
            for focus, value in (('continue', 35), ('restart', 215)):
                # Slightly different images in separate sessions avoid hash leakage.
                pixels = bytes((value + index, value + index, value + index, 0)) * 1920 * 1080
                frame = _png(1920, 1080, pixels)
                name = f'{split}-{focus}.png'
                (self.root / name).write_bytes(frame)
                samples.append({'session_id': split, 'split': split, 'frame_path': name,
                                'frame_sha256': hashlib.sha256(frame).hexdigest(),
                                'scene': 'pause', 'focus': focus,
                                'label_provenance': {'kind': 'human_review', 'reviewed': True,
                                                     'reviewer': 'synthetic-test', 'reviewed_at': '2026-09-27',
                                                     'review_ref': 'synthetic fixture, not gameplay evidence'}})
        self.manifest = {'schema': 'menu-model-reviewed-manifest-v1', 'resolution': [1920, 1080],
                         'game_build_id': 'test-build', 'language': 'en', 'samples': samples}
        self.manifest_path = self.root / 'manifest.json'
        self.model_path = self.root / 'candidate.json'
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def fit(self):
        return train_manifest(self.manifest_path, self.model_path, epochs=40)

    def test_actual_weights_roundtrip_and_no_automatic_deployment(self):
        report = self.fit()
        self.assertTrue(report['training_performed'])
        self.assertFalse(report['runtime_deployed'])
        self.assertEqual(report['evaluation']['test']['wrong_accepted'], 0)
        self.assertEqual(report['evaluation']['test']['accepted'], 2)
        model = MenuClassifier.load(self.model_path)
        self.assertTrue(any(v != 0 for row in model.weights for v in row))
        pixels, w, h = read_diagnostic_png(self.root / 'test-continue.png')
        self.assertEqual(model.predict(pixels, w, h, game_build_id='test-build', language='en').abstain_reason,
                         'unapproved_checkpoint')
        prediction = model.predict(pixels, w, h, game_build_id='test-build', language='en', require_approved=False)
        self.assertEqual(prediction.label, 'pause:continue')
        self.assertGreaterEqual(prediction.elapsed_ms, 0)
        with self.assertRaises(FileExistsError):
            self.fit()

    def test_approval_binds_checkpoint_and_scope(self):
        report = self.fit()
        approval_path = self.root / 'approval.json'
        approval = {'schema': 'menu-model-approval-v1', 'checkpoint_sha256': report['checkpoint_sha256'],
                    'approved': True, 'reviewer': 'synthetic-test', 'reviewed_at': '2026-09-27',
                    'review_ref': 'test-only approval'}
        approval_path.write_text(json.dumps(approval), encoding='utf-8')
        model = MenuClassifier.load(self.model_path, approval_path=approval_path)
        pixels, w, h = read_diagnostic_png(self.root / 'test-continue.png')
        self.assertIsNone(model.predict(pixels, w, h, game_build_id='test-build', language='en').abstain_reason)
        self.assertEqual(model.predict(pixels, w, h, game_build_id='different-build', language='en').abstain_reason,
                         'scope_mismatch')
        self.assertEqual(model.predict(pixels, w, h, game_build_id='test-build', language='zh').abstain_reason,
                         'scope_mismatch')
        self.assertEqual(model.predict(pixels, w // 2, h, game_build_id='test-build', language='en').abstain_reason,
                         'invalid_frame')
        self.model_path.write_bytes(self.model_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Approval'):
            MenuClassifier.load(self.model_path, approval_path=approval_path)

    def test_unknown_visual_content_abstains(self):
        self.fit()
        model = MenuClassifier.load(self.model_path)
        # Red is far outside the reviewed grayscale samples, even if softmax wins.
        pixels = bytes((0, 0, 255, 0)) * 1920 * 1080
        prediction = model.predict(pixels, 1920, 1080, game_build_id='test-build', language='en', require_approved=False)
        self.assertIsNone(prediction.label)
        self.assertIn(prediction.abstain_reason, ('low_confidence', 'low_margin', 'outside_training_support'))

    def test_malformed_or_nonfinite_checkpoint_rejected_before_inference(self):
        self.fit()
        checkpoint = json.loads(self.model_path.read_bytes())
        for malformed in ([], {'schema': 'invalid'}):
            with self.assertRaises(ValueError):
                MenuClassifier(malformed)
        for key, value in (('labels', None), ('weights', [None]), ('thresholds', None)):
            with self.subTest(key=key):
                bad = dict(checkpoint)
                bad[key] = value
                with self.assertRaises(ValueError):
                    MenuClassifier(bad)
        checkpoint['weights'][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'matrix'):
            MenuClassifier(checkpoint)

    def test_automatic_ocr_labels_rejected(self):
        self.manifest['samples'][0]['label_provenance']['kind'] = 'ocr'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'review provenance'):
            self.fit()
        self.assertFalse(self.model_path.exists())

    def test_session_leakage_rejected(self):
        self.manifest['samples'][2]['session_id'] = 'train'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Session leakage'):
            self.fit()

    def test_hash_tampering_and_duplicate_frames_rejected(self):
        first, second = self.manifest['samples'][:2]
        first['frame_sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.fit()
        first['frame_sha256'] = second['frame_sha256']
        first['frame_path'] = second['frame_path']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate frame hash'):
            self.fit()


if __name__ == '__main__':
    unittest.main()
