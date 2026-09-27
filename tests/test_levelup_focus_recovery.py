from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from playmodel.games.brotato import session
from playmodel.games.brotato.menu import BUTTONS, ButtonSelection, selected_button
from playmodel.games.brotato.menu_focus import LevelUpFocusRecovery
from playmodel.games.brotato.neural_menu_controller import MenuDirective
from test_brotato_session_recovery import frame, ocr


class LevelUpFocusTests(unittest.TestCase):
    def missing(self):
        return ButtonSelection(None, None, 'no_selected_candidate',
                               tuple((f'choose_{i}', .144) for i in range(4)))

    def observe(self, recovery, selection=None, **kwargs):
        values = dict(scene='level_up', decision_id='d', frame_id='one', observed_at_ns=100, now_ns=200)
        values.update(kwargs)
        return recovery.observe(selection or self.missing(), **values)

    def test_two_independent_frames_one_nonconfirming_key_then_stop(self):
        r = LevelUpFocusRecovery()
        self.assertEqual(self.observe(r), 'wait')
        self.assertEqual(self.observe(r), 'wait')
        self.assertEqual(self.observe(r, frame_id='two', observed_at_ns=110), 'left')
        r.mark_sent('d')
        self.assertIsNone(self.observe(r, frame_id='three', observed_at_ns=120))

    def test_ambiguous_blank_stale_and_other_scene_remain_blocked(self):
        for selection, kwargs in [
            (ButtonSelection(None,None,'ambiguous_selection', self.missing().ratios), {}),
            (ButtonSelection(None,None,'no_selected_candidate'), {}),
            (self.missing(), {'scene':'loot'}),
            (self.missing(), {'now_ns':800_000_200}),
        ]:
            with self.subTest(kwargs=kwargs, reason=selection.reason):
                self.assertIsNone(self.observe(LevelUpFocusRecovery(), selection, **kwargs))

    def test_actual_failure_is_absent_focus_not_an_ambiguous_highlight(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        p = Path(__file__).resolve().parents[1]/('artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/'
            'evaluation-0-source-fb30a6a954fb4abdb544fcce34d543ce/segments/20260927T083406Z-c1261331/'
            'menus/20260927T083443Z-fab556b2/frame.png')
        if not p.exists():
            self.skipTest('local original game evidence unavailable')
        pixels,w,h = read_diagnostic_png(p)
        selection = selected_button(pixels,w,h,scene='level_up')
        self.assertEqual(selection.reason, 'no_selected_candidate')
        r = LevelUpFocusRecovery()
        self.assertEqual(self.observe(r,selection), 'wait')
        self.assertEqual(self.observe(r,selection,frame_id='two',observed_at_ns=110), 'left')

    def run_session_fixture(self, *, deferred=False):
        # Visible gray button lettering, but no selected gray background.
        missing = frame(*[(x+45,645,x+80,677) for x in (52,421,790,1159)])
        self.assertEqual(selected_button(missing,1920,1080,scene='level_up').reason,'no_selected_candidate')
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            frames = iter(([(frame(BUTTONS['level_up']['choose_2']),'level_up')]*5 if deferred else
                           [(missing,'level_up'),(missing,'level_up'),
                            (frame(BUTTONS['level_up']['choose_2']),'level_up')]) + [(frame(),'result')])
            capture, reader, controller, neural = Mock(), Mock(), Mock(), Mock()
            controller.hwnd=1
            controller._context.key_state.return_value=0
            neural.recorder.model.config.context_dim=16
            neural.handle.return_value=MenuDirective('navigate','choose_2','decision','test')
            if deferred:
                neural.authorize_enter.side_effect=[False,False,False,False,True]
            reports={}
            def read(output, **kwargs):
                pixels,scene=next(frames)
                target=output/str(len(reports))
                target.mkdir(parents=True)
                reports[str(target/'frame.png')]=ocr(scene)
                now=time.perf_counter_ns()
                return ({'session_directory':str(target),'hwnd':1,'frame_sha256':str(len(reports))*64,
                         'capture_started_at_ns':now,'available_at_ns':now},pixels,1920,1080)
            capture.read.side_effect=read
            reader.read.side_effect=lambda path:reports[str(path)]
            for name,value in [('MenuCapture',Mock(return_value=capture)),('MenuOcr',Mock(return_value=reader)),
                ('BackgroundController',Mock(return_value=controller)),
                ('capture_session',Mock(side_effect=OSError('offline'))),
                ('inspect_installation',Mock(return_value={'installations':[]}))]:
                stack.enter_context(patch.object(session,name,value))
            report=session.run_session.__wrapped__(root/'game.exe',root/'sessions',waves=1,seconds=10,
                stop_file=root/'STOP',ocr_script=root/'unused.ps1',edit=False,
                combat_runner=Mock(side_effect=AssertionError('unexpected combat')),neural_menu=neural)
            self.assertEqual(report['reason'],'run_finished')
            self.assertEqual([c.args[0] for c in controller.tap_menu.call_args_list],['enter'] if deferred else ['left','enter'])
            self.assertEqual(len(reports),6 if deferred else 4)
            self.assertEqual(neural.authorize_enter.call_count,5 if deferred else 1)
            neural.mark_sent.assert_called_once()
            actions=json.loads((Path(report['session_directory'])/'menu-actions.json').read_text(encoding='utf-8'))
            if not deferred:
                self.assertEqual(actions[0]['policy_origin'],'verified_navigation_recovery')
                self.assertFalse(actions[0]['learned_choice'])

    def test_session_reobserves_after_acquisition_before_confirming(self):
        self.run_session_fixture()

    def test_deferred_authorization_does_not_count_as_repeated_input(self):
        self.run_session_fixture(deferred=True)
