import json
import os
from pathlib import Path
import tempfile
import unittest

from playmodel.learning_video_titles import label_completed


class LearningVideoTitleTests(unittest.TestCase):
    def test_closed_evaluation_has_honest_title_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'artifacts/recurrent-cycles/cycle/evaluation-0-candidate-id'
            segment = run / 'segments/session'
            segment.mkdir(parents=True)
            (run / 'run-operation.json').write_text(json.dumps({'split': 'evaluation', 'partial': False}))
            (segment / 'report.json').write_text('{}')
            media = root / 'media/captures/session'
            (media / 'original').mkdir(parents=True)
            source = media / 'original/raw.mp4'
            source.write_bytes(b'original movie bytes')
            manifest = media / 'recording.json'
            manifest.write_text(json.dumps({'output_path': str(source), 'closed': False}))
            self.assertEqual(label_completed(root), [])
            manifest.write_text(json.dumps({'output_path': str(source), 'closed': True}))
            result = label_completed(root)
            self.assertEqual(len(result), 1)
            self.assertIn('후보모델평가', result[0]['title'])
            self.assertNotIn('HP', result[0]['title'])
            self.assertTrue(os.path.samefile(source, result[0]['display_path']))
            self.assertEqual(source.read_bytes(), b'original movie bytes')
            self.assertEqual(json.loads(manifest.read_text(encoding='utf-8'))['output_path'], str(source))
            self.assertEqual(label_completed(root), [])
            self.assertTrue((root / 'media/learning-videos.html').is_file())

    def test_recovery_never_claims_training_and_outside_source_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'artifacts/recurrent-cycles/cycle/partial-recovery-id'
            segment = run / 'segments/session'
            segment.mkdir(parents=True)
            (run / 'run-operation.json').write_text(json.dumps({'split': 'evaluation', 'partial': True}))
            (segment / 'report.json').write_text('{}')
            media = root / 'media/captures/session'
            (media / 'original').mkdir(parents=True)
            outside = root / 'outside.mp4'
            outside.write_bytes(b'keep')
            manifest = media / 'recording.json'
            manifest.write_text(json.dumps({'output_path': str(outside), 'closed': True}))
            self.assertEqual(label_completed(root), [])
            source = media / 'original/raw.mp4'
            source.write_bytes(b'keep')
            manifest.write_text(json.dumps({'output_path': str(source), 'closed': True}))
            result = label_completed(root)
            self.assertEqual(result[0]['purpose'], '복구-학습평가제외')


if __name__ == '__main__':
    unittest.main()
