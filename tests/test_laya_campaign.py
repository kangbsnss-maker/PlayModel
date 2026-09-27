from pathlib import Path
import tempfile
import unittest
from playmodel.laya.campaign import LearningCampaign, CONCEPTS


class CampaignTests(unittest.TestCase):
    def test_progress_survives_restart_without_double_counting_or_claiming_requested_character(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'campaign.json'
            campaign = LearningCampaign(path)
            self.assertEqual(campaign.current()['character_slot'], 1)
            self.assertFalse(campaign.record({'run_id': 'interrupted', 'full_run_complete': False}))
            report = {'run_id': 'finished', 'full_run_complete': True,
                      'setup_conditions': {'character_slot': 3, 'character': 'observed character', 'weapons': ['observed weapon']}}
            self.assertTrue(campaign.record(report))
            restored = LearningCampaign(path)
            self.assertFalse(restored.record(report))
            self.assertEqual(restored.current()['character_slot'], 2)
            self.assertFalse(restored.state['attempts'][0]['character_request_matched'])
            self.assertEqual(sum(restored.state['actual_combinations'].values()), 1)

    def test_rotation_covers_calibrated_slots_weapon_positions_and_concepts(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = LearningCampaign(Path(directory) / 'campaign.json')
            for cursor, slot, weapon, concept in [(49, 50, '@rotate:1', CONCEPTS[1]),
                    (50, 1, '@rotate:1', CONCEPTS[0]), (600, 1, '@rotate:0', CONCEPTS[1]),
                    (1200, 1, '@rotate:0', CONCEPTS[2]), (1800, 1, '@rotate:0', CONCEPTS[0])]:
                campaign.state['cursor'] = cursor
                actual = campaign.current()
                self.assertEqual((actual['character_slot'], actual['weapon'], actual['concept']), (slot, weapon, concept))
            first=[]
            combinations=set()
            for cursor in range(1800):
                campaign.state['cursor']=cursor
                item=campaign.current()
                combinations.add((item['character_slot'],item['weapon'],item['concept']))
                if cursor<3:first.append(item)
            self.assertEqual(len(combinations),1800)
            self.assertEqual(len({item['concept'] for item in first}),3)
            self.assertEqual(len({item['weapon'] for item in first}),3)
