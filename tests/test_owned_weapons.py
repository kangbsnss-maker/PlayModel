from copy import deepcopy
import unittest
import time
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from playmodel.games.brotato.owned_weapons import verify_operation, same,OwnedWeaponLearning,inventory_equal
from playmodel.games.brotato.menu import navigation_key,BUTTONS
from playmodel.games.brotato.setup_run import focused_tile


class OwnedWeaponTests(unittest.TestCase):
    def test_first_owned_input_creates_macro_directory_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run'/'macro-actions'
            inventory=OwnedWeaponLearning(SimpleNamespace(output_directory=output))
            inventory.last_key=('right','recover_unowned_hover',{'observed_at_ns':100})
            inventory.sent(101,102)
            inventory.sent(103,104)
            rows=[json.loads(line) for line in (output/'owned-inputs.jsonl').read_text().splitlines()]
            self.assertEqual([row['sent_at_ns'] for row in rows],[102,104])
            self.assertIsNone(rows[0]['decision_id'])

    def test_single_owned_weapon_focus_does_not_require_a_runner_up(self):
        pixels=bytes([200,200,200,255])*100*100
        self.assertEqual(focused_tile(pixels,100,{0:(0,0,96,96)}),0)
        self.assertIsNone(focused_tile(pixels,100,{}))
        self.assertIsNone(focused_tile(bytes(len(pixels)),100,{0:(0,0,96,96)}))

    def test_restart_closes_hover_before_reading_occluded_inventory(self):
        menu=SimpleNamespace(pending_decision=None,client=Mock())
        inventory=OwnedWeaponLearning(menu)
        now=time.perf_counter_ns()
        shot={'capture_started_at_ns':now,'session_directory':'.','frame_sha256':'source'}
        with patch('playmodel.games.brotato.owned_weapons.snapshot',return_value=None), \
             patch('playmodel.games.brotato.owned_weapons.popup',return_value={'selected':None}):
            self.assertEqual(inventory.step(shot,{'available_at_ns':now},b''),'right')
        self.assertEqual(inventory.last_key[1],'recover_unowned_hover')
        menu.client.choose.assert_not_called()
        menu.client.accept.assert_not_called()

    def sample(self,time,icons=('knife','sling'),currency=37):
        return {'wave':1,'count':len(icons),'icons':list(icons),'currency':currency,
                'observed_at_ns':time,'available_at_ns':time+1,
                'target_identity':{'hwnd':1,'pid':2,'executable':'game'}}

    def test_recycle_requires_matching_payout_and_exact_target_removal(self):
        before=self.sample(10)
        after=self.sample(30,('sling',),40)
        second=self.sample(40,('sling',),40)
        self.assertTrue(verify_operation(before,after,second,action='recycle',slot=0,payout=3,sent_at_ns=20))
        for field,value in [('currency',39),('icons',['knife']),('wave',2),('observed_at_ns',19)]:
            wrong=deepcopy(second);wrong[field]=value
            self.assertFalse(verify_operation(before,after,wrong,action='recycle',slot=0,payout=3,sent_at_ns=20))

    def test_combine_is_not_sale_or_arbitrary_count_drop(self):
        before=self.sample(10,('knife','knife','sling'))
        after=self.sample(30,('knife','sling'))
        second=self.sample(40,('knife','sling'))
        self.assertTrue(verify_operation(before,after,second,action='combine',slot=0,payout=3,sent_at_ns=20))
        self.assertFalse(verify_operation(before,after,second,action='recycle',slot=0,payout=3,sent_at_ns=20))
        before['icons'][1]='other'
        self.assertFalse(verify_operation(before,after,second,action='combine',slot=0,payout=3,sent_at_ns=20))

    def test_cancel_preserves_inventory_and_never_labels_tier(self):
        before=self.sample(10)
        self.assertTrue(verify_operation(before,self.sample(30),self.sample(40),action='cancel',slot=0,payout=3,sent_at_ns=20))
        self.assertFalse(same(self.sample(30),self.sample(30)))

    def test_last_weapon_is_not_recycled(self):
        self.assertFalse(verify_operation(self.sample(10,('knife',)),self.sample(30,(),40),
            self.sample(40,(),40),action='recycle',slot=0,payout=3,sent_at_ns=20))

    def test_unverified_result_returns_to_stable_shop_without_credit(self):
        menu=SimpleNamespace(pending_decision=None,client=Mock())
        inventory=OwnedWeaponLearning(menu)
        inventory.before=self.sample(10)
        inventory.after=self.sample(30,('other',),50)
        inventory.card={'payout':3}
        inventory.decision={'decision_id':'pending','action_id':'recycle'}
        inventory.sent_at=20
        inventory.steps=80
        seen=self.sample(40,('other',),50)
        with patch('playmodel.games.brotato.owned_weapons.snapshot',return_value=seen), \
             patch('playmodel.games.brotato.owned_weapons.popup',return_value=None):
            self.assertIsNone(inventory.step({'capture_started_at_ns':time.perf_counter_ns()}, {}, b''))
        menu.client.discard.assert_called_once_with('pending','owned_result_unverified_stable_shop')
        menu.client.accept.assert_not_called()
        self.assertIn(1,inventory.done)

    def test_stale_pair_cannot_verify_inventory_change(self):
        self.assertFalse(same(self.sample(10),self.sample(2_000_000_020)))

    def test_overall_budget_also_bounds_hover_and_forced_cancel(self):
        menu=SimpleNamespace(pending_decision=None,client=Mock())
        inventory=OwnedWeaponLearning(menu)
        inventory.before=self.sample(10)
        inventory.decision={'decision_id':'pending'}
        inventory.sent_at=20
        inventory.steps=120
        inventory.force_cancel=True
        inventory.opened_card_key=('old',)
        with patch('playmodel.games.brotato.owned_weapons.snapshot',return_value=None), \
             patch('playmodel.games.brotato.owned_weapons.popup',return_value={'selected':None}):
            self.assertEqual(inventory.step({'capture_started_at_ns':time.perf_counter_ns()}, {}, b''),'wait')
        self.assertIsNone(inventory.decision)
        self.assertIsNone(inventory.opened_card_key)
        self.assertFalse(inventory.force_cancel)
        menu.client.discard.assert_called_once()

    def test_icon_antialias_tolerance_is_bounded_and_rejects_sparse_masks(self):
        before=self.sample(10,('a',))
        after=self.sample(30,('b',))
        mask=(1<<200)-1
        before['icon_masks']=[hex(mask)]
        after['icon_masks']=[hex(mask^3)]
        self.assertTrue(inventory_equal(before,after,[0]))
        after['icon_masks']=[hex(mask^7)]
        self.assertFalse(inventory_equal(before,after,[0]))
        before['icon_masks']=['0x3'];after['icon_masks']=['0x1']
        self.assertFalse(inventory_equal(before,after,[0]))

    def test_weapon_to_go_route_is_right_for_both_shop_layouts(self):
        for slot in range(3):
            for rect in (BUTTONS['shop']['depart'],(1484,981,1884,1048)):
                self.assertEqual(navigation_key(BUTTONS['shop'][f'owned_{slot}'],rect,scene='shop'),'right')
