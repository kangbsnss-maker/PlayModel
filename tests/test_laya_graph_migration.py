import copy
import hashlib
import unittest

from playmodel.laya.graph_migration import validate_boundary_fix_migration
from playmodel.laya.records import canonical


class GraphMigrationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {'laya/worker.py':
            '8bf18a20ea5965c6dfada5f4054568876a578e63426d71a80b2911b00fe980e4',
            'laya/forward.py': 'unchanged', 'laya/visual.py': 'same-preprocessing'}
        fingerprint = hashlib.sha256(canonical(self.sources).encode()).hexdigest()
        self.bundle = {'graph_hash': fingerprint,
                       'report': {'graph_hash': fingerprint, 'graph_sources': self.sources}}
        self.current = {**self.sources, 'laya/worker.py': 'new-boundary-fix'}

    def test_only_known_worker_change_can_preserve_weights(self):
        result = validate_boundary_fix_migration(self.bundle, self.current)
        self.assertTrue(result['weights_preserved'])
        self.assertTrue(result['behavior_version_changes'])

    def test_preprocessing_change_rejected(self):
        self.current['laya/visual.py'] = 'different'
        with self.assertRaises(ValueError):
            validate_boundary_fix_migration(self.bundle, self.current)

    def test_unknown_parent_and_tampered_manifest_rejected(self):
        for key in ('graph_hash',):
            bundle = copy.deepcopy(self.bundle)
            bundle[key] = 'wrong'
            with self.assertRaises(ValueError):
                validate_boundary_fix_migration(bundle, self.current)
        self.sources['laya/worker.py'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_boundary_fix_migration(self.bundle, self.current)
