import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from playmodel.video import edit_plan, timestamp, SessionRecording, SCENE, SOURCE
from playmodel.games.brotato.style import load_style
from playmodel.games.brotato.session import shop_observation, upgrade_choice
from playmodel.games.brotato.menu import ENGLISH_ANCHORS, classify_scene


def ocr(words):
    return {'lines': [{'words': [{'text': text, 'x': x, 'y': y, 'width': width, 'height': 25}
                                for text, x, y, width in words]}]}


class VideoTests(unittest.TestCase):
    def test_transient_windows_manifest_lock_retries_atomic_replace(self):
        import json
        import os
        import threading
        with tempfile.TemporaryDirectory() as folder:
            recorder=SessionRecording.__new__(SessionRecording)
            recorder.directory=Path(folder)
            recorder._lock=threading.Lock()
            recorder.report={'started':True}
            recorder.events=[]
            original=os.replace
            calls=[]
            def replace(source,target):
                calls.append(target)
                if len(calls)==1: raise PermissionError('temporary reader lock')
                return original(source,target)
            with patch('playmodel.video.os.replace',side_effect=replace), patch('playmodel.video.time.sleep'):
                recorder._persist()
            self.assertEqual(len(calls),2)
            self.assertTrue(json.loads((Path(folder)/'recording.json').read_text())['started'])

    def test_preexisting_recording_is_never_stopped(self):
        class Existing:
            def __init__(self): self.calls = []
            def call(self, name, **kwargs):
                self.calls.append(name)
                if name == 'GetRecordStatus': return {'outputActive': True}
                raise AssertionError(name)
            def close(self): pass
        client = Existing()
        with tempfile.TemporaryDirectory() as folder, patch('playmodel.video.ObsClient', return_value=client):
            with self.assertRaisesRegex(OSError, 'ownership refused'):
                SessionRecording(Path(folder), max_seconds=20)
        self.assertEqual(client.calls, ['GetRecordStatus'])

    def test_async_start_and_stop_are_confirmed_before_restoring_directory(self):
        class Delayed:
            def __init__(self):
                self.state, self.polls, self.directory = 'idle', 0, 'old-directory'
                self.calls = []
            def call(self, name, **kwargs):
                self.calls.append(name)
                if name == 'GetRecordStatus':
                    self.polls += 1
                    if self.state == 'starting' and self.polls >= 2: self.state = 'active'
                    if self.state == 'stopping' and self.polls >= 2: self.state = 'idle'
                    return {'outputActive': self.state in ('active','stopping'), 'outputPaused': False, 'outputDuration': 0}
                if name == 'GetStreamStatus': return {'outputActive': False}
                if name == 'GetCurrentProgramScene': return {'currentProgramSceneName': SCENE}
                if name == 'GetSceneItemList': return {'sceneItems': [{'sceneItemEnabled': True, 'sourceName': SOURCE,'inputKind':'game_capture'}]}
                if name == 'GetInputSettings': return {'inputSettings': {'capture_mode':'window','window':'Brotato:Engine:Brotato.exe'}}
                if name == 'GetInputList': return {'inputs': []}
                if name == 'GetRecordDirectory': return {'recordDirectory': self.directory}
                if name == 'SetRecordDirectory':
                    if kwargs['recordDirectory'] == 'old-directory':
                        assert self.state == 'idle', 'Must finalize output before restoring settings'
                    self.directory = kwargs['recordDirectory']
                if name == 'StartRecord': self.state,self.polls = 'starting',0
                if name == 'StopRecord':
                    self.state,self.polls = 'stopping',0
                    return {'outputPath':'test.mp4'}
                return {}
            def close(self): pass
        client = Delayed()
        with tempfile.TemporaryDirectory() as folder, patch('playmodel.video.ObsClient', return_value=client), patch('playmodel.video.subprocess.Popen'):
            recorder = SessionRecording(Path(folder), max_seconds=20)
            result = recorder.close()
        self.assertEqual(result['output_path'], 'test.mp4')
        self.assertEqual(client.directory, 'old-directory')
        self.assertEqual(client.calls.count('StopRecord'), 1)

    def test_cut_mapping_bounds_and_subtitle_overlap(self):
        events = [{'at_ns': int(t*1e9), 'kind': kind, 'text': kind} for t, kind in
                  [(2, 'session_start'), (3, 'menu_choice'), (25, 'combat_start'), (50, 'stop'), (60, 'outside')]]
        cues, cuts = edit_plan(events, 0, 52)
        self.assertEqual(cuts, [(0., 8.), (23., 35.), (48., 52)])
        self.assertEqual(cues[0]['end'], 3)
        self.assertEqual(cues[-1]['end'], 52)
        self.assertEqual(timestamp(3661.125), '01:01:01,125')
        self.assertEqual(timestamp(-1), '00:00:00,000')

    def test_english_anchors_and_wrong_geometry(self):
        for scene, anchors in ENGLISH_ANCHORS.items():
            report = ocr([(token, region[0]+5, region[1]+5, 120) for token, region in anchors])
            self.assertEqual(classify_scene(report).scene, scene)
        report = ocr([('Shop', 20, 50, 100), ('GO', 1550, 850, 100)])
        self.assertEqual(classify_scene(report).scene, 'shop')
        report['lines'][0]['words'][1]['y'] = 500
        self.assertEqual(classify_scene(report).scene, 'unknown')

    def test_currency_icons_never_count_as_digits(self):
        report = ocr([('Shop', 25, 45, 95), ('Wave', 135, 45, 100), ('4', 269, 45, 20),
                      ('0', 762, 50, 35), ('223', 816, 50, 100), ('4', 1360, 51, 20)])
        self.assertEqual(shop_observation(report)['currency'], 223)
        self.assertEqual(shop_observation(report)['reroll_cost'], 4)
        report['lines'][0]['words'][-1]['text'] = '4?'
        self.assertIsNone(shop_observation(report)['reroll_cost'])

    def test_style_weights_do_not_confuse_melee_and_generic_damage(self):
        report = ocr([('+4MeleeDamage', 40, 510, 250), ('+2Armor', 410, 510, 250),
                      ('+3RangedDamage', 780, 510, 250), ('+5Damage', 1150, 510, 250)])
        self.assertEqual(upgrade_choice(report), 2)
        style = load_style()
        style['stat_weights']['armor'] = 30
        self.assertEqual(upgrade_choice(report, style), 1)


if __name__ == '__main__':
    unittest.main()
