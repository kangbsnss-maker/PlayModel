import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from playmodel.laya.combat_economy import CombatEconomy, fixed_evaluation_eligible
from playmodel.laya.records import digest
from playmodel.learning.outcome_values import OutcomeValues


class EconomyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.economy=CombatEconomy(self.root/'models')
        self.economy.begin_run('run')

    def evidence(self):
        frame=self.root/'frame'
        frame.write_bytes(b'pixels')
        terminal=self.root/'terminal.json'
        terminal.write_text(json.dumps({'kind':'wave_clear','verified':True,
            'independent_of_policy':True,'observed_at_ns':500,'frame_ref':str(frame),
            'frame_sha256':digest(frame)}))
        actions=self.root/'actions.jsonl'
        actions.write_text('')
        report=self.root/'report.json'
        report.write_text(json.dumps({'run_id':'run','verified_terminal_boundary':True,
            'recorder_complete':True,'worker_stopped':True,'actions_path':str(actions),
            'actions_sha256':digest(actions)}))
        return {'path':str(terminal),'sha256':digest(terminal),'report_path':str(report),
                'report_sha256':digest(report),'run_id':'run'}

    def purchase(self):
        source=self.root/'choice.json'
        source.write_text('{}')
        record={'decision_id':'d','evidence':{'run_id':'run'},'state':{'wave':1},
                'options':{'buy:0':'Weapon SMG'},'action_id':'buy:0','_source_path':str(source)}
        self.economy.record_purchase(record,{'game_application_verified':True,'verified_at_ns':100})

    def test_confirmed_purchase_to_later_outcome_learns_and_reloads(self):
        self.purchase()
        before=self.economy.model_hash
        result=self.economy.finish('wave_clear',self.evidence())
        self.assertNotEqual(before,result['model_hash'])
        self.assertEqual(result['curriculum']['economic_examples'],1)
        restored=CombatEconomy(self.root/'models')
        self.assertEqual(restored.model_hash,result['model_hash'])
        self.assertIsNone(result['summary']['kills'])
        with self.assertRaisesRegex(ValueError,'already consumed'):
            self.economy.finish('wave_clear',self.evidence())

    def test_future_purchase_not_credited(self):
        self.purchase()
        self.economy.history[0]['verified_at_ns']=700
        result=self.economy.finish('wave_clear',self.evidence())
        self.assertEqual(result['curriculum']['economic_examples'],0)

    def test_factorized_dataset_trains_persists_and_freezes(self):
        from test_evasion_dataset import EvasionDatasetTests
        evidence = self.evidence()
        rows = EvasionDatasetTests().rows()
        for row in rows:
            row.update(successful_transport_reported=True, frame_ref='frame',
                       frame_sha256=digest(self.root/'frame'))
        actions = self.root/'actions.jsonl'
        actions.write_text('\n'.join(json.dumps(r) for r in rows))
        report_path = Path(evidence['report_path'])
        report = json.loads(report_path.read_text())
        report['actions_sha256'] = digest(actions)
        report_path.write_text(json.dumps(report))
        evidence['report_sha256'] = digest(report_path)
        terminal = Path(evidence['path'])
        content = json.loads(terminal.read_text())
        content['observed_at_ns'] = 2000000000
        terminal.write_text(json.dumps(content))
        evidence['sha256'] = digest(terminal)
        original = self.economy.model_hash
        self.economy.finish('wave_clear', evidence, learn=False)
        self.assertEqual(original, self.economy.model_hash)
        self.assertEqual(list(self.economy.directory.glob('evasion-*.jsonl')), [])
        result = self.economy.finish('wave_clear', evidence)
        self.assertEqual(result['curriculum']['evasion_prediction_examples'], 1)
        self.assertNotEqual(original, self.economy.model_hash)
        state = json.loads(Path(result['record']).read_text())
        dataset = state['evasion_dataset']
        self.assertEqual(digest(dataset['path']), dataset['sha256'])
        self.assertEqual(CombatEconomy(self.economy.directory).model_hash, self.economy.model_hash)

    def test_new_run_does_not_inherit_old_purchases(self):
        self.purchase()
        self.economy.begin_run('next')
        self.assertEqual(self.economy.history,[])

    def test_fixed_evaluation_does_not_train_or_write_revision(self):
        self.purchase()
        before=self.economy.model_hash
        result=self.economy.finish('wave_clear',self.evidence(),learn=False)
        self.assertEqual(before,self.economy.model_hash)
        self.assertEqual(result['model_updates'],0)
        self.assertEqual(list((self.root/'models').glob('revision-*.json')),[])

    def test_recovered_or_changed_configuration_cannot_be_evaluation(self):
        valid={'full_run_complete':True,'scheduling_recovery_count':0,'choice_updates':[]}
        self.assertTrue(fixed_evaluation_eligible(valid,{'version':'1'},{'version':'1'}))
        for change in ({'scheduling_recovery_count':1}, {'observation_gaps':[{}]},
                       {'choice_updates':[{'status':'excluded'}]}, {'full_run_complete':False}):
            self.assertFalse(fixed_evaluation_eligible({**valid,**change},{'version':'1'},{'version':'1'}))
        self.assertFalse(fixed_evaluation_eligible(valid,{'version':'1'},{'version':'2'}))

    def test_gap_abandons_delayed_credit(self):
        self.purchase()
        self.economy.abandon()
        result=self.economy.finish('wave_clear',self.evidence())
        self.assertEqual(result['curriculum']['economic_examples'],0)

    def test_values_learn_opposite_observed_returns(self):
        values=OutcomeValues()
        for _ in range(100): values.fit([({'choice':'a'},1.),({'choice':'b'},-1.)])
        self.assertGreater(values.predict({'choice':'a'}),values.predict({'choice':'b'}))

    def test_purchase_ranking_changes_with_combat_context(self):
        values=OutcomeValues()
        for _ in range(500):
            values.fit([({'choice':'weapon','combat:pressure':20},1.),
                        ({'choice':'passive','combat:pressure':20},-1.),
                        ({'choice':'weapon','combat:pressure':-20},-1.),
                        ({'choice':'passive','combat:pressure':-20},1.)])
        for pressure, better, worse in [(20,'weapon','passive'),(-20,'passive','weapon')]:
            self.assertGreater(values.predict({'choice':better,'combat:pressure':pressure}),
                               values.predict({'choice':worse,'combat:pressure':pressure}))

    def test_emergency_override_removes_laya_credit(self):
        try:
            from playmodel.games.brotato.tactical_runtime import TacticalCombatActor
        except ImportError:
            self.skipTest('optional torch')
        import time
        now=time.perf_counter_ns()
        situation={'world':{'valid':True,'available_at_ns':now,'fresh_until_ns':now+1000000000},
                   'state':{},'options':{'retreat':'retreat'},'signature':'sig'}
        session=SimpleNamespace(error=None,offer=Mock(),resolve=Mock(return_value={'action_id':'retreat'}))
        actor=TacticalCombatActor(session,planner=SimpleNamespace(observe=Mock(return_value=situation)))
        actor.experience=SimpleNamespace(observe=Mock(return_value={}),emergency=Mock(return_value=(3,'fast_collision_hypothesis')))
        frame=SimpleNamespace(metadata={'capture_started_at_ns':now,'sample_width':100,'sample_height':100},available_at_ns=now)
        with patch('playmodel.games.brotato.tactical_state.execute_tactic',return_value=1):
            packet=actor.propose(frame,object())
        self.assertEqual(packet.action,3)
        self.assertIsNone(packet.decision)
        self.assertEqual(packet.fallback_reason,'fast_collision_hypothesis')
