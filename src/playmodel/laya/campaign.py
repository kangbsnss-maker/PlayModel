"""Durable training rotation; requested coverage is never observed coverage."""
import json
from pathlib import Path
from playmodel.atomic_io import atomic_json

CONCEPTS = ('balanced survival and progress', 'survival and recovery', 'damage and resource collection')


class LearningCampaign:
    def __init__(self, path):
        self.path = Path(path)
        self.state = (json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else
            {'schema': 'playmodel.learning-campaign.v1', 'cursor': 0, 'completed_run_ids': [],
             'actual_combinations': {}, 'attempts': []})
        if self.state.get('schema') != 'playmodel.learning-campaign.v1' or type(self.state.get('cursor')) is not int:
            raise ValueError('Invalid learning campaign state')

    def current(self):
        cursor = self.state['cursor']
        return {'index': cursor, 'character_slot': 1 + cursor % 50,
                'weapon': '@rotate:' + str((cursor % 50 + cursor // 50) % 12),
                'concept': CONCEPTS[(cursor % 50 + cursor // 600) % len(CONCEPTS)],
                'schedule': 'interleaved_character_weapon_concept_v2',
                'coverage_scope': 'calibrated_slots_and_observed_available_weapons_not_all_unlocks'}

    def record(self, report):
        run_id = report.get('run_id')
        if report.get('full_run_complete') is not True or not run_id:
            return False
        if run_id in self.state['completed_run_ids']:
            return False
        requested = self.current()
        actual = report.get('setup_conditions') or {}
        key = json.dumps({'character': actual.get('character', 'unknown'),
                          'weapons': actual.get('weapons') or [], 'concept': requested['concept']},
                         sort_keys=True, ensure_ascii=False)
        self.state['actual_combinations'][key] = self.state['actual_combinations'].get(key, 0) + 1
        self.state['completed_run_ids'].append(run_id)
        self.state['attempts'].append({'run_id': run_id, 'requested': requested, 'actual': actual,
            'character_request_matched': actual.get('character_slot') == requested['character_slot'],
            'concept_semantics': 'conditioning_intent_not_verified_achieved_build'})
        self.state['cursor'] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.state, durable=True)
        return True
