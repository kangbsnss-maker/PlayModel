"""Isolated, build-conditioned conservative movement Q candidate.

This trainer has no game input API and cannot replace the live Laya actor.
Recorded actions are off-policy experience, not imitation targets.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from playmodel.laya.records import canonical, digest
from .screen_replay import load_replay
from .visual_decision import VisualDecisionContext

SCHEMA='playmodel.offline-survival-cql.v1'
CONTEXT_SIZE=128


def context_vector(context):
    """Fixed signed lexical features; no pretrained semantic understanding claim."""
    result=torch.zeros(CONTEXT_SIZE)
    for key in ('character','weapons','concept','difficulty','traits'):
        value=context.get(key)
        text=json.dumps(value,ensure_ascii=False).casefold() if value else '<unknown>'
        for word in re.findall(r'\w+|<unknown>',text):
            token=hashlib.sha256((key+':'+word).encode()).digest()
            result[int.from_bytes(token[:4],'little')%CONTEXT_SIZE]+=1 if token[4]%2 else -1
    return result/result.norm().clamp_min(1)


class SurvivalQ(nn.Module):
    def __init__(self):
        super().__init__()
        visual=VisualDecisionContext(1)
        self.encoder=visual.encoder
        self.temporal=visual.temporal
        self.context=nn.Sequential(nn.Linear(CONTEXT_SIZE,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Linear(128,9))

    def forward(self,frames,context):
        if frames.dtype!=torch.uint8 or frames.ndim!=5 or frames.shape[2:]!=(3,96,96):
            raise ValueError('Expected uint8 RGB screen histories')
        batch,count=frames.shape[:2]
        encoded=self.encoder(frames.reshape(-1,3,96,96).float()/255)
        history,_=self.temporal(encoded.reshape(batch,count,-1))
        return self.head(torch.cat((history[:,-1],self.context(context)),dim=1))


def conservative_loss(q,actions,rewards,discounts,next_online,next_target,alpha=1.,weights=None):
    """Double DQN target, terminal discount zero, CQL discrete regularizer."""
    with torch.no_grad():
        best=next_online.argmax(1,keepdim=True)
        target=rewards+discounts*next_target.gather(1,best).squeeze(1)
    actual=q.gather(1,actions[:,None]).squeeze(1)
    weights=torch.ones_like(actual) if weights is None else weights
    bellman=(F.smooth_l1_loss(actual,target,reduction='none')*weights).mean()
    conservative=((torch.logsumexp(q,dim=1)-actual)*weights).mean()
    return bellman+alpha*conservative,bellman,conservative


def train(manifest_path,output,*,epochs=8,batch_size=32,threads=2,seed=7):
    if not 1<=epochs<=10000 or not 1<=batch_size<=512 or not 1<=threads<=16:
        raise ValueError('Invalid offline training budget')
    output=Path(output).resolve()
    if output.exists():raise ValueError('Candidate output must be new')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    manifest,frames=load_replay(manifest_path)
    rows=manifest['rows']
    split={name:[i for i,r in enumerate(rows) if r['split']==name] for name in ('train','validation','test')}
    if not split['train'] or not split['validation']:
        raise ValueError('Separate training and validation runs required')
    output.mkdir(parents=True)
    contexts=torch.stack([context_vector(r['build_context']) for r in rows])
    build_keys=[canonical(r['build_context']) for r in rows]
    build_counts=Counter(build_keys[i] for i in split['train'])
    training_indices=set(split['train'])
    # Every recorded build contributes equal total weight per replay pass.
    weights=torch.tensor([len(split['train'])/(len(build_counts)*build_counts[build_keys[i]])
        if i in training_indices else 1. for i in range(len(rows))])
    states=torch.tensor([r['state'] for r in rows]);following=torch.tensor([r['next_state'] for r in rows])
    actions=torch.tensor([r['action'] for r in rows])
    rewards=torch.tensor([r['reward'] for r in rows],dtype=torch.float32)
    discounts=torch.tensor([r['discount'] for r in rows],dtype=torch.float32)
    model=SurvivalQ();target=copy.deepcopy(model).eval()
    target.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    updates=0;sample_visits=0;started=time.monotonic();history=[]

    def loss(indices):
        q=model(frames[states[indices]],contexts[indices])
        with torch.no_grad():
            next_online=model(frames[following[indices]],contexts[indices])
            next_target=target(frames[following[indices]],contexts[indices])
        return conservative_loss(q,actions[indices],rewards[indices],discounts[indices],next_online,next_target,
                                 weights=weights[indices] if model.training else None)

    def evaluate(indices):
        if not indices:return None
        model.eval();total=0.
        with torch.no_grad():
            for offset in range(0,len(indices),batch_size):
                batch=indices[offset:offset+batch_size]
                total+=float(loss(batch)[0])*len(batch)
        return total/len(indices)

    for epoch in range(epochs):
        model.train();total=0.
        order=torch.tensor(split['train'])[torch.randperm(len(split['train']))].tolist()
        for offset in range(0,len(order),batch_size):
            indices=order[offset:offset+batch_size]
            objective,_,_=loss(indices)
            if not torch.isfinite(objective):raise ValueError('Nonfinite offline loss; candidate rejected')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1;sample_visits+=len(indices)
            if updates%100==0:target.load_state_dict(model.state_dict())
            total+=float(objective.detach())*len(indices)
        history.append({'epoch':epoch+1,'train_objective':total/len(order),
                        'validation_objective':evaluate(split['validation'])})
        (output/'progress.json').write_text(canonical({'epochs_completed':epoch+1,
            'optimizer_steps':updates,'sample_visits':sample_visits,'completed_games_added':0,
            'latest':history[-1]}),encoding='utf8')
    report={'schema':SCHEMA,'manifest':{'path':str(Path(manifest_path).resolve()),'sha256':digest(manifest_path)},
            'device':'cpu','threads':threads,'seed':seed,'epochs':epochs,'optimizer_steps':updates,
            'sample_visits':sample_visits,'unique_training_transitions':len(split['train']),
            'split_counts':{k:len(v) for k,v in split.items()},'completed_games_added':0,
            'elapsed_seconds':time.monotonic()-started,'history':history,
            'test_objective':evaluate(split['test']),
            'parameter_max_change':max(float((v-initial[k]).abs().max()) for k,v in model.state_dict().items()),
            'build_coverage':dict(Counter(canonical(r['build_context']) for r in rows)),
            'build_balancing':'inverse_training_build_frequency_loss_weight',
            'initialization':'random_no_parent_holdout_leakage',
            'reward':manifest['reward'],'cql_alpha':1.,'learning_rate':1e-4,
            'required_executor_schema':manifest['required_executor_schema'],
            'target_sync_steps':100,'live_promotion_allowed':False,
            'live_survival_improvement_verified':False,
            'scope':'wave_local_movement_starting_build_conditioned_not_purchase_or_full_run_growth'}
    report['trainer_sources']=[{'path':str(p),'sha256':digest(p)} for p in
        (Path(__file__),Path(__file__).with_name('screen_replay.py'),Path(__file__).with_name('visual_decision.py'))]
    checkpoint=output/'candidate.pt'
    torch.save({'schema':SCHEMA,'state_dict':model.state_dict(),'report':report},checkpoint)
    report['checkpoint']={'path':str(checkpoint),'sha256':digest(checkpoint)}
    (output/'report.json').write_text(canonical(report),encoding='utf8')
    return report
