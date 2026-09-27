import tempfile
from pathlib import Path
import unittest

from playmodel.games.brotato.motion import MovementStabilizer
from playmodel.games.brotato.vision import VisionObservation,VisualCandidate
from playmodel.learning import MOVEMENTS
from playmodel.learning.worker import child_file,database,enqueue


class BackgroundLearningTests(unittest.TestCase):
    def test_death_without_title_requires_both_body_and_confirmation_regions(self):
        from playmodel.games.brotato.menu import classify_scene
        from playmodel.games.brotato.session import calibrated_rules
        words=[{'text':'KiIIed by:','x':876,'y':365,'width':166,'height':40},
               {'text':'Ok','x':936,'y':671,'width':50,'height':30}]
        report={'text':'KiIIed by: Ok','lines':[{'words':[word]} for word in words]}
        rules=calibrated_rules()
        self.assertEqual(classify_scene(report).scene,'death')
        self.assertEqual(rules.classify(report['text'],lines=report['lines'],image_size=(1920,1080)),'death')
        words[1]['y']=850
        self.assertEqual(classify_scene(report).scene,'unknown')
        self.assertIsNone(rules.classify(report['text'],lines=report['lines'],image_size=(1920,1080)))
        words[1]['y']=671;words[0]['y']=850
        self.assertEqual(classify_scene(report).scene,'unknown')

    def test_death_font_variant_requires_positioned_anchors(self):
        from playmodel.games.brotato.menu import classify_scene
        from playmodel.games.brotato.session import calibrated_rules
        words=[{'text':'RUN LOST','x':732,'y':164,'width':456,'height':69},
               {'text':'KiIIed by:','x':876,'y':365,'width':166,'height':40}]
        report={'text':'RUN LOST KiIIed by:', 'lines':[{'words':[w]} for w in words]}
        self.assertEqual(classify_scene(report).scene,'death')
        rules=calibrated_rules()
        self.assertEqual(rules.classify(report['text'],lines=report['lines'],image_size=(1920,1080)),'death')
        words[1]['y']=900
        self.assertEqual(classify_scene(report).scene,'unknown')
        self.assertIsNone(rules.classify(report['text'],lines=report['lines'],image_size=(1920,1080)))

    def test_direction_commitment_and_emergency_turn(self):
        right=MOVEMENTS.index((1,0))
        state=VisionObservation(navigation=(1.,0.,True),player=(.5,.5))
        stabilizer=MovementStabilizer(200)
        mask=stabilizer.mask(state,right,1_000_000_000)
        self.assertEqual(sum(mask),1)
        self.assertTrue(mask[right])
        reversed_state=VisionObservation(navigation=(-1.,0.,True),player=(.5,.5))
        changed=stabilizer.mask(reversed_state,right,1_010_000_000)
        self.assertFalse(changed[right])
        self.assertTrue(changed[MOVEMENTS.index((-1,0))])

    def test_receding_threat_does_not_cancel_direction_hold(self):
        right=MOVEMENTS.index((1,0))
        stabilizer=MovementStabilizer(200)
        def state(x):
            return VisionObservation(navigation=(1.,.4,True),player=(.5,.5),
                hazards=(VisualCandidate(x,.5,.01,1.,'hazard','test'),))
        held=stabilizer.mask(state(.46),right,1_000_000_000)
        self.assertEqual(sum(held),1)
        self.assertTrue(held[right])
        approaching=stabilizer.mask(state(.54),right,1_010_000_000)
        self.assertGreater(sum(approaching),1)

    def test_queue_is_idempotent_and_artifacts_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'episode.json').write_text('{}',encoding='utf-8')
            queue=root/'queue.sqlite3'
            first=enqueue(queue,root,style_sha='test')
            self.assertEqual(first,enqueue(queue,root,style_sha='test'))
            with database(queue) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
            with self.assertRaises(ValueError): child_file(root,'../outside.json')

    def test_real_character_focus_calibration(self):
        from playmodel.games.brotato.setup_run import CHARACTER_GRID,focused_tile
        from playmodel.games.brotato.capture import read_diagnostic_png
        root=Path(__file__).resolve().parents[1]/'data/raw/brotato-observations'
        for token,expected in [('20260926T175012Z-5462b8ba',0),('20260926T175037Z-d9309da2',1)]:
            path=root/token/'frame.png'
            if not path.exists(): self.skipTest('Private calibration frames excluded from Git')
            pixels,w,h=read_diagnostic_png(path)
            self.assertEqual(focused_tile(pixels,w,CHARACTER_GRID),expected)
        from playmodel.games.brotato.menu import BUTTONS
        path=root/'20260926T175206Z-9fb3f6d1/frame.png'
        if path.exists():
            pixels,w,h=read_diagnostic_png(path)
            self.assertEqual(focused_tile(pixels,w,{i:BUTTONS['difficulty'][f'danger_{i}'] for i in range(7)}),6)

    def test_real_colored_stat_focus_and_levelup_panel(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        from playmodel.games.brotato.menu import selected_stat_row
        from playmodel.games.brotato.vision import BrotatoVision
        root=Path(__file__).resolve().parents[1]/'artifacts/brotato-sessions'
        green=root/'20260926T180358Z-46d6cfc4/menus/20260926T180400Z-d0afaf10/frame.png'
        levelup=root/'20260926T175925Z-b9445e8d/pilots/20260926T175928Z-f9071437/phase-frame.png'
        if not green.exists() or not levelup.exists(): self.skipTest('Private calibration frames excluded')
        pixels,w,h=read_diagnostic_png(green)
        self.assertIsNotNone(selected_stat_row(pixels,w,h).selected_id)
        pixels,w,h=read_diagnostic_png(levelup)
        self.assertEqual(BrotatoVision().observe(pixels,w,h).status,'menu_panel_layout')


if __name__=='__main__': unittest.main()
