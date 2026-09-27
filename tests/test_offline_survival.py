import copy
import math
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from playmodel.learning.screen_replay import group_split, transition_values, interrupted_interval,terminal_bridge


class ReplayTests(unittest.TestCase):
    def test_terminal_bridge_allows_only_verified_scene_release_not_new_input(self):
        first={'sent_at_ns':100,'target_identity':{},'sequence':1,'generation':1}
        phase={'metadata':{'capture_started_at_ns':110},'available_at_ns':120,
               'decision':'abstain','sequence':2,'vision':{'combat_likely':False,'rejected_at_ns':130}}
        events=[{'kind':'authority','reason':'screen_changed','at_ns':140},
                {'kind':'release','reason':'screen_changed','at_ns':160,'send_started_at_ns':150,
                 'send_finished_at_ns':160,'deadline_ns':170,'receipt':{'transmitted':True,'acknowledged':None}}]
        bridge=terminal_bridge(first,200,phase,events)
        self.assertEqual(bridge['observed_alive_seconds'],0.)
        for extra in [{'kind':'proposed','at_ns':180},{'kind':'authority','reason':'ai_granted','at_ns':180},
                      {'kind':'release','reason':'human','at_ns':180}]:
            self.assertIsNone(terminal_bridge(first,200,phase,[*events,extra]))
        failed=copy.deepcopy(events);failed[-1]['receipt']['transmitted']=False
        self.assertIsNone(terminal_bridge(first,200,phase,failed))
        self.assertIsNone(terminal_bridge(first,200,phase,[*events,
            {'kind':'release','reason':'human','send_started_at_ns':190,'at_ns':250}]))
        self.assertIsNone(terminal_bridge(first,6_000_000_000,phase,events))

    def test_release_attempt_crossing_next_frame_invalidates_interval(self):
        events=[{'kind':'release','send_started_at_ns':150,'at_ns':250}]
        self.assertTrue(interrupted_interval(events,100,200))
        self.assertFalse(interrupted_interval(events,260,300))

    def pair(self):
        first={'run_id':'run','epoch':'epoch','generation':1,'target_identity':{},
               'execution_id':'a','sent_at_ns':1_000_000_000,'held_after':[68],
               'frame_sha256':'a'}
        second={**first,'execution_id':'b','previous_execution_id':'a','sequence':2,
                'observed_at_ns':1_100_000_000,'transport_started_at_ns':1_120_000_000,
                'held_before':[68],'frame_sha256':'b'}
        first['sequence']=1
        return first,second

    def test_gaps_changed_authority_and_unexecuted_chains_are_censored(self):
        first,second=self.pair()
        self.assertIsNotNone(transition_values(first,second))
        for key,value in [('epoch','other'),('generation',2),('held_before',[]),
                          ('previous_execution_id','missing'),('observed_at_ns',2_000_000_000)]:
            with self.subTest(key=key):
                bad={**second,key:value}
                self.assertIsNone(transition_values(first,bad))

    def test_time_partition_invariance_and_terminal_has_no_invented_alive_time(self):
        first,second=self.pair()
        result=transition_values(first,second)
        half=60*(1-math.exp(-.05/60))
        self.assertAlmostEqual(result['reward'],half+math.exp(-.05/60)*half)
        terminal=transition_values(first,None,terminal='death',terminal_at=1_100_000_000)
        self.assertEqual((terminal['reward'],terminal['discount'],terminal['observed_alive_seconds']),(-5.,0.,0.))
        self.assertIsNone(transition_values(first,None,terminal='death',terminal_at=999_999_999))
        self.assertIsNone(transition_values(first,None,terminal='death',terminal_at=3_000_000_000))

    def test_identical_screens_do_not_earn_alive_reward(self):
        first,second=self.pair();second['frame_sha256']='a'
        self.assertEqual(transition_values(first,second)['reward'],0.)

    def test_split_is_run_grouped(self):
        assignments=[group_split('run-'+str(i)) for i in range(100)]
        self.assertEqual(set(assignments),{'train','validation','test'})
        self.assertEqual(assignments,[group_split('run-'+str(i)) for i in range(100)])


class OfflineLossTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch environment required')
        self.torch=torch

    def test_freeze_preserves_terminal_excludes_late_input_and_rejects_missing_terminal(self):
        import test_tactical_receipts as fixtures
        from playmodel.learning.screen_replay import freeze_replay,load_replay
        from playmodel.laya.records import digest
        fixture=fixtures.TacticalReceiptTests()
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root
        pilot=root/'session'/'run'/'segments'/'segment'/'pilots'/'pilot'
        pilot.mkdir(parents=True)
        def save(path,value):
            path.write_text(json.dumps(value),encoding='utf8')
            return {'path':str(path),'sha256':digest(path)}
        receipts=[];proofs=[];attempts=[]
        for number in (1,2,3):
            r=fixture.receipt(number,'native_transition' if number==1 else 'existing_owned_hold')
            frame=Path(r['frame_ref']);frame.write_bytes(bytes([number,0,0,255])*4)
            r['frame_sha256']=digest(frame);r['verified_at_ns']=r['sent_at_ns']
            save(frame.with_suffix('.json'),{'capture_started_at_ns':r['observed_at_ns'],
                **fixture.identity,'sample_width':2,'sample_height':2})
            if receipts:
                r.update(previous_execution_id=receipts[-1]['execution_id'],
                    previous_receipt_path=proofs[-1]['path'],previous_receipt_sha256=proofs[-1]['sha256'],
                    ownership_receipt_path=proofs[0]['path'],ownership_receipt_sha256=proofs[0]['sha256'])
            proofs.append(save(pilot/f'receipt-{number}.json',r));receipts.append(r)
            attempts.append({'generation':r['generation'],'sequence':r['sequence'],'transmitted':True,
                'error':None,'transport_finished_at_ns':r['sent_at_ns'],'background_stages':r['background_stages']})
        terminal_at=receipts[2]['observed_at_ns']-1
        terminal_source=pilot/'terminal-source.png';terminal_source.write_bytes(b'test terminal source')
        terminal={'frame_ref':str(terminal_source),'frame_sha256':digest(terminal_source)}
        save(pilot/'observation.json',{**fixture.identity,'capture_started_at_ns':terminal_at,
                                       'frame_sha256':digest(terminal_source)})
        save(pilot/'terminal.json',terminal)
        ledger=pilot/'actions.jsonl';ledger.write_text('\n'.join(json.dumps(r) for r in receipts))
        save(pilot/'input-attempts.json',attempts)
        lifecycle=[{'kind':'authority','reason':'ai_granted','at_ns':1},
                   {'kind':'authority','reason':'closed','at_ns':terminal_at+10}]
        save(pilot/'control-events.json',lifecycle)
        character_directory=root/'character';character_directory.mkdir()
        character_source=character_directory/'frame.png';character_source.write_bytes(b'fixture character')
        save(character_directory/'observation.json',{**fixture.identity,'capture_started_at_ns':1,
            'frame_sha256':digest(character_source)})
        # Older setup records omitted the hash/time fields; original capture
        # metadata must supply both, never a guessed time or unchecked hash.
        save(pilot.parents[3]/'setup-evidence.json',{'setup':{'context':{
            'character':'Brawler','character_source':str(character_source),'weapons':['Fist'],'concept':'melee'}}})
        save(pilot/'report.json',{'split':'train','verified_terminal_boundary':True,'recorder_complete':True,
            'worker_stopped':True,'tactical_collection_eligible':True,'terminal_kind':'death',
            'run_id':'run','tactical_epoch':'epoch','actions_path':str(ledger),'actions_sha256':digest(ledger),
            'tactical_receipts':proofs})
        with patch('playmodel.learning.screen_replay.verified_outcome',return_value=(-1,terminal_at,terminal)):
            path=freeze_replay(root,[root/'session'],root/'replay')
            manifest,frames=load_replay(path)
            self.assertEqual(len(manifest['rows']),2)
            self.assertTrue(manifest['rows'][-1]['done'])
            self.assertEqual(manifest['excluded']['post_terminal_input'],1)
            save(pilot/'control-events.json',[*lifecycle,{'kind':'release',
                'send_started_at_ns':receipts[1]['sent_at_ns']+1,'at_ns':terminal_at+1}])
            with self.assertRaisesRegex(ValueError,'no_valid_terminal_transition'):
                freeze_replay(root,[root/'session'],root/'rejected')
            with self.assertRaisesRegex(ValueError,'source changed'):
                load_replay(path)
            report=json.loads((pilot/'report.json').read_text())
            report['tactical_receipts']=[];save(pilot/'report.json',report)
            with self.assertRaisesRegex(ValueError,'no_executed_actions'):
                freeze_replay(root,[root/'session'],root/'empty-actions')

    def test_target_is_detached_double_dqn_and_terminal_zero_bootstrap(self):
        from playmodel.learning.offline_survival import conservative_loss
        t=self.torch
        q=t.zeros(2,9,requires_grad=True)
        online=t.zeros(2,9,requires_grad=True);target=t.ones(2,9,requires_grad=True)
        with t.no_grad():online[:,3]=10;target[:,3]=2;target[:,4]=100
        result,bellman,cql=conservative_loss(q,t.tensor([0,1]),t.tensor([-5.,0.]),
            t.tensor([0.,.5]),online,target)
        self.assertAlmostEqual(float(bellman),2.5)
        self.assertAlmostEqual(float(cql),math.log(9),places=5)
        result.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNone(online.grad);self.assertIsNone(target.grad)

    def test_build_context_changes_model_input_without_future_inventory(self):
        from playmodel.learning.offline_survival import context_vector
        t=self.torch
        a={'character':'Brawler','weapons':['Fist'],'concept':'melee'}
        b={'character':'Ranger','weapons':['Pistol'],'concept':'ranged'}
        self.assertFalse(t.equal(context_vector(a),context_vector(b)))
        self.assertTrue(t.equal(context_vector(a),context_vector({**a,'future_loot':'extra'})))
        self.assertTrue(t.isfinite(context_vector({})).all())


if __name__=='__main__':
    unittest.main()
