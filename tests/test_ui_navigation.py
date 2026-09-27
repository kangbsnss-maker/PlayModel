import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from playmodel.learning.ui_navigation import UiNavigationMemory


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'graph.jsonl'
        self.memory = UiNavigationMemory(self.path,'build1')
        self.t = time.perf_counter_ns()-1000000000

    def shot(self, offset):
        directory = self.root / str(offset)
        directory.mkdir(exist_ok=True)
        raw = b'frame'+str(offset).encode()
        (directory/'frame.png').write_bytes(raw)
        return {'session_directory': str(directory), 'frame_sha256': hashlib.sha256(raw).hexdigest(),
                'capture_started_at_ns': self.t+offset, 'available_at_ns': self.t+offset+1,
                'hwnd':1,'pid':2,'executable':'game.exe'}

    def test_one_left_wrap_confirmed_and_reloaded(self):
        self.assertEqual(self.memory.propose('difficulty','danger_0','danger_6','right'),'left')
        self.memory.sent('difficulty','danger_0','left',self.shot(1),self.t+5)
        self.assertFalse(self.memory.observe('difficulty','danger_6',self.shot(8)))
        self.assertTrue(self.memory.observe('difficulty','danger_6',self.shot(12)))
        restored = UiNavigationMemory(self.path,'build1')
        self.assertEqual(restored.counts[('difficulty','danger_0','left')], {'danger_6':1})
        self.assertEqual(restored.propose('difficulty','danger_0','danger_6','right'),'left')
        self.assertFalse(UiNavigationMemory(self.path,'different_build').counts)

    def test_contradictory_hint_disabled_and_enter_never_generated(self):
        self.memory._count({'scene':'difficulty','from':'danger_0','key':'left','to':'danger_0'})
        self.assertEqual(self.memory.propose('difficulty','danger_0','danger_6','right'),'right')
        self.assertEqual(self.memory.propose('difficulty','danger_0','danger_6','enter'),'enter')

    def test_shop_observed_shortcut_beats_longer_path(self):
        for source,key,dest in [('buy_3','down','depart'),('buy_3','right','other'),('other','down','depart')]:
            self.memory._count({'scene':'shop','from':source,'key':key,'to':dest})
        self.assertEqual(self.memory.propose('shop','buy_3','depart','right'),'down')

    def test_before_send_or_changed_identity_not_accepted(self):
        self.memory.sent('shop','buy_3','down',self.shot(1),self.t+5)
        self.assertFalse(self.memory.observe('shop','depart',self.shot(2)))
        shot = self.shot(10)
        shot['pid']=9
        with self.assertRaises(ValueError): self.memory.observe('shop','depart',shot)

    def test_saved_graph_cannot_inject_enter(self):
        self.path.write_text(json.dumps({'scope':'build1','kind':'observed_transition',
            'scene':'shop','from':'buy_3','to':'depart','key':'enter'})+'\n')
        with self.assertRaisesRegex(ValueError,'non-direction'):
            UiNavigationMemory(self.path,'build1')

    def test_fixed_graph_observes_but_does_not_learn(self):
        self.memory.learning=False
        self.memory.sent('difficulty','danger_0','left',self.shot(1),self.t+5)
        self.memory.observe('difficulty','danger_6',self.shot(8))
        self.memory.observe('difficulty','danger_6',self.shot(12))
        self.assertFalse(self.path.exists())
        self.assertFalse(self.memory.counts)
