"""Read-only learning evidence counts and loopback preference mutation boundary."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('laya_dashboard',ROOT/'scripts/laya_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = dashboard.Dashboard(self.root)
        self.worker = self.root/'artifacts/laya-learning/laya-fixture/laya'

    def choice(self,identifier,domain='menu'):
        value = {'decision_id':identifier,'decision_domain':domain,'behavior_version':'v1',
                 'state':{'hp':'unknown'},'options':{'retreat':'Retreat'},'action_id':'retreat',
                 'distribution':{'retreat':1.0}}
        path = self.worker/f'choice-{identifier}.json'
        save(path,value)
        return path

    def update(self,identifier,decisions):
        files = []
        for decision_id in decisions:
            path = self.choice(decision_id,'combat_tactic' if decision_id=='tactic' else 'menu')
            accepted = self.worker/f'accepted-{decision_id}.json'
            save(accepted,{'accepted':True})
            for proof in (path,accepted):
                files.append({'path':str(proof),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()})
        directory = self.worker/f'update-{identifier}'
        outcome = {'origin':'local_detector','verified':True,'independent_of_policy':True,'kind':'death'}
        terminal = directory/'terminal.json'
        terminal_sha = save(terminal,outcome)
        ledger = directory/'actions.jsonl'
        ledger.write_text('{"execution_id":"actual"}\n',encoding='utf8')
        files.append({'path':str(ledger),'sha256':hashlib.sha256(ledger.read_bytes()).hexdigest()})
        manifest_hash = save(directory/'dataset-manifest.json',{'schema':'playmodel.laya-outcome-dataset.v1',
            'source_version':'v1','split':'train','files':files})
        save(directory/'report.json',{'accepted':True,'status':'updated','decisions':decisions,
            'source_version':'v1','behavior_version':'v2-'+identifier,'dataset_manifest_sha256':manifest_hash,
            'optimizer_steps':1,'created_utc':'2026-09-27T00:00:00+00:00',
            'head_hash_before':'before','head_hash_after':'after','encoder_hash_before':'same','encoder_hash_after':'same',
            'kl_per_choice':[.001]*len(decisions),'verified_outcome':outcome,
            'outcome':{'kind':'death','terminal_path':str(terminal),'terminal_sha256':terminal_sha}})

    def test_counts_deduplicate_trained_choices_and_exclude_abandoned_pending(self):
        self.update('a',['menu','tactic','abandoned'])
        self.update('b',['menu'])
        self.update('c',['tactic'])
        self.choice('pending')
        save(self.worker/'abandoned-case.json',{'accepted':['abandoned'],'pending':[]})
        run = self.worker.parent/'run'
        save(run/'cycle-run.json',{'run_id':'run','full_run_complete':True,
             'schema':'playmodel.laya-run.v1','split':'train','menu_choice_backend':'local_laya'})
        save(self.worker.parent/'recovery/cycle-run.json',{'run_id':'recovery','full_run_complete':True,'recovery_only':True})
        status = self.app.status(refresh=True)
        self.assertEqual(status['counts']['trained_decisions'],2)
        self.assertEqual(status['counts']['menu_decisions'],3)
        self.assertEqual(status['counts']['tactic_decisions'],1)
        self.assertEqual(status['counts']['runs_completed'],1)
        self.assertEqual(status['counts']['accepted_updates'],2)
        self.assertEqual(len(status['errors']),1)

    def test_copied_version_is_not_counted_twice_and_synthetic_outcome_excluded(self):
        self.update('a',['one'])
        report = json.loads((self.worker/'update-a/report.json').read_text())
        directory = self.worker/'update-copy'
        save(directory/'report.json',report)
        self.assertEqual(self.app.status(refresh=True)['counts']['accepted_updates'],1)
        report['verified_outcome']['origin']='synthetic_fixture'
        save(self.worker/'update-a/report.json',report)
        save(directory/'report.json',report)
        self.assertEqual(self.app.status(refresh=True)['counts']['accepted_updates'],0)

    def test_corrupt_training_proof_is_not_valid_learning_data(self):
        self.update('a',['one'])
        save(self.worker/'accepted-one.json',{'tampered':True})
        status = self.app.status(refresh=True)
        self.assertEqual(status['counts']['trained_decisions'],0)
        self.assertEqual(status['counts']['accepted_updates'],0)
        self.assertTrue(status['errors'])

    def test_atomic_preferences_revision_reason_and_linked_decision_journal(self):
        self.choice('abc123')
        result = self.app.save_preferences({'revision':0,'values':{'retreat':1.5},'reason':'관측에 대한 사람 메모',
                                            'related_decision_id':'abc123'})
        self.assertEqual(result['preferences']['revision'],1)
        self.assertEqual(result['application'],'next_safe_boundary')
        records = list(self.app.journal.glob('request-*.json'))
        self.assertEqual(len(records),1)
        saved = json.loads(records[0].read_text(encoding='utf8'))
        self.assertEqual(saved['related_decision_id'],'abc123')
        self.assertFalse(saved['reward_source'])
        self.assertEqual(len(list(self.app.journal.glob('committed-*.json'))),1)
        with self.assertRaises(dashboard.Conflict):
            self.app.save_preferences({'revision':0,'values':{},'reason':'stale'})

    def test_invalid_feedback_never_changes_preferences(self):
        for update in ({'reason':''},{'values':{'retreat':2.01}},{'values':{'retreat':float('nan')}},
                       {'values':{'arbitrary':1}},{'related_decision_id':'../outside'},
                       {'related_decision_id':'unknown'}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                self.app.save_preferences({'revision':0,'values':{},'reason':'test',**update})
        self.assertFalse(self.app.preference_path.exists())


class DashboardHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        page = self.root/'docs/guides/laya-dashboard.html'
        page.parent.mkdir(parents=True)
        page.write_text('<script nonce="__CSP_NONCE__">window.test=true</script>',encoding='utf8')
        self.server = dashboard.create_server(self.root)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.origin = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self,method,path,body=None,headers=None):
        conn = http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        conn.request(method,path,body=body,headers=headers or {})
        response = conn.getresponse()
        result = response.status,dict(response.getheaders()),response.read().decode('utf8')
        conn.close()
        return result

    def test_health_and_same_origin_nonce_page(self):
        status,headers,body = self.request('GET','/api/health')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['service'],dashboard.SERVICE)
        self.assertNotIn('Access-Control-Allow-Origin',headers)
        status,headers,body = self.request('GET','/')
        self.assertEqual(status,200)
        self.assertNotIn('__CSP_NONCE__',body)
        self.assertIn(self.server.app.nonce,headers['Content-Security-Policy'])
        self.assertEqual(self.request('GET','/../../configs/local/laya-preferences.json')[0],404)

    def test_host_origin_and_csrf_all_required(self):
        self.assertEqual(self.request('GET','/api/status',headers={'Host':'evil.example'})[0],403)
        body = json.dumps({'revision':0,'values':{'retreat':1},'reason':'observed risk'})
        headers = {'Content-Type':'application/json','Origin':self.origin,'X-CSRF-Token':self.server.app.csrf}
        for bad in ({'Origin':'https://evil.example'},{'X-CSRF-Token':'wrong'},
                    {'Host':'evil.example'},{'Sec-Fetch-Site':'cross-site'}):
            with self.subTest(bad=bad):
                self.assertEqual(self.request('POST','/api/preferences',body,{**headers,**bad})[0],403)
        self.assertFalse(self.server.app.preference_path.exists())
        self.assertEqual(self.request('POST','/api/preferences',body,headers)[0],200)
        self.assertEqual(self.request('POST','/api/preferences',body,headers)[0],409)

    def test_large_or_non_json_request_rejected(self):
        headers = {'Content-Type':'text/plain','Origin':self.origin,'X-CSRF-Token':self.server.app.csrf}
        self.assertEqual(self.request('POST','/api/preferences','{}',headers)[0],400)
        headers['Content-Type']='application/json'
        self.assertEqual(self.request('POST','/api/preferences','x'*17000,headers)[0],400)


if __name__ == '__main__':
    unittest.main()
