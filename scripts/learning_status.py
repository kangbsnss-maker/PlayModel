"""Read-only local service, training queue and campaign status."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import json
from pathlib import Path
from playmodel.learning.worker import database

root=Path(__file__).resolve().parents[1]
result={}
for name,relative in [('campaign','artifacts/campaign/status.json'),('learner','artifacts/learning/worker-status.json'),
                      ('editor','artifacts/media-worker-status.json')]:
    path=root/relative
    if path.exists(): result[name]=json.loads(path.read_text(encoding='utf-8'))
queue=root/'artifacts/learning/queue.sqlite3'
if queue.exists():
    with database(queue) as db:
        result['jobs_by_state']=dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
        result['recent_jobs']=[{'state':state,'error':error,'result':json.loads(raw) if raw else None}
                               for state,error,raw in db.execute('SELECT state,error,result FROM jobs ORDER BY updated DESC LIMIT 2')]
print(json.dumps(result,ensure_ascii=True,indent=2))
