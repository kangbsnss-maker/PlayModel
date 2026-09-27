import unittest
from types import SimpleNamespace
from playmodel.games.brotato.combat_experience import CombatExperience, bar_crop


def frame(now):
    return SimpleNamespace(metadata={'sample_width': 100, 'sample_height': 100},
                           pixels=bytes([20,20,20,255])*10000, sequence=now)


def situation(now, tracks=None):
    return {'options': {'avoid_crossing': 'avoid'}, 'world': {'valid': True,
        'observed_at_ns': now, 'available_at_ns': now+1, 'fresh_until_ns': now+250000000,
        'player': [.5,.5], 'pickups': [], 'stats': {}, 'tracks': tracks or [],
        'aspect_ratio': 1., 'parameters': {'spacing': .1}}}


def track(identity=1, position=.6):
    return {'track_id': identity, 'position': [position,.5], 'relative': [position-.5,0],
            'radius': .02, 'velocity': [-.1,0], 'pattern_hypotheses': [],
            'collision_time_hypothesis_seconds': .1}


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.vision = SimpleNamespace(camera_status='unknown', camera_shift=None, hp_fill_fraction=.8)

    def test_missing_object_is_censored_not_kill_or_pickup_reward(self):
        memory = CombatExperience()
        memory.observe(frame(100), situation(100,[track()]), self.vision)
        second = memory.observe(frame(100000100), situation(100000100), self.vision)
        self.assertEqual(second['missing_tracks'], [1])
        self.assertFalse(second['reward_eligible'])
        self.assertIsNone(second['numeric_enemy_hp'])
        self.assertFalse(second['map_boundary_verified'])

    def test_motion_pairs_use_previous_input_future_target_only(self):
        memory = CombatExperience()
        memory.observe(frame(100), situation(100,[track()]), self.vision)
        second = memory.observe(frame(100000100), situation(100000100,[track(position=.59)]), self.vision)
        pair = second['motion_pairs'][0]
        self.assertLess(pair['input_observed_at_ns'], pair['target_observed_at_ns'])
        self.assertAlmostEqual(pair['target_velocity'][0], -.1)
        gap = memory.observe(frame(2000000000), situation(2000000000,[track()]), self.vision)
        self.assertEqual(gap['motion_pairs'], [])

    def test_screen_edge_alone_never_means_wall(self):
        memory = CombatExperience()
        for i in range(10):
            value = situation(100+i*10000000)
            value['world']['player'] = [.99,.5]
            sample = memory.observe(frame(i), value, self.vision)
        self.assertEqual(max(sample['movement_not_observed']), 0)

    def test_stale_emergency_cannot_send(self):
        memory = CombatExperience()
        current = situation(100,[track()])
        self.assertIsNone(memory.emergency(current, {}, 999999999))

    def test_hp_crop_absent_stays_unknown(self):
        value = bar_crop(frame(100), [.5,.5], .02)
        self.assertIsNone(value['ratio'])
        self.assertIsNone(value['numeric_hp'])

    def test_object_branch_zero_effect_then_learns_crop_encoder(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch')
        from playmodel.learning.visual_decision import VisualDecisionContext
        torch.set_num_threads(1)
        model = VisualDecisionContext(16)
        images = torch.randint(0,256,(1,2,3,96,96),dtype=torch.uint8)
        crops = torch.randint(0,256,(1,2,3,32,32),dtype=torch.uint8)
        candidates = torch.randn(1,3,16)
        torch.testing.assert_close(model(images,candidates), model(images,candidates,crops), rtol=0,atol=0)
        initial = model.object_encoder[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(images,candidates,crops),torch.tensor([1])).backward()
            optimizer.step()
        self.assertFalse(torch.equal(initial,model.object_encoder[0].weight))
