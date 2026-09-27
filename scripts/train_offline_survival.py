"""Freeze verified movement experience or train an isolated local CQL candidate."""
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze')
    freeze.add_argument('--session',action='append',required=True)
    freeze.add_argument('--output',required=True)
    freeze.add_argument('--per-segment',type=int,default=64)
    freeze.add_argument('--max-segments',type=int)
    fit=sub.add_parser('train')
    fit.add_argument('--manifest',required=True)
    fit.add_argument('--output',required=True)
    fit.add_argument('--epochs',type=int,default=8)
    fit.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.command=='freeze':
        import torch
        torch.set_num_threads(2)
        from playmodel.learning.screen_replay import freeze_replay
        path=freeze_replay(Path(__file__).resolve().parents[1],args.session,args.output,
                          per_segment=args.per_segment,max_segments=args.max_segments)
        result=json.loads(path.read_text(encoding='utf8'))
        print(json.dumps({'manifest':str(path),'transitions':result['unique_transitions'],
                          'splits':result['split_counts'],'excluded':result['excluded']}))
    else:
        from playmodel.learning.offline_survival import train
        report=train(args.manifest,args.output,epochs=args.epochs,threads=args.threads)
        print(json.dumps({k:report[k] for k in ('optimizer_steps','sample_visits','elapsed_seconds',
            'parameter_max_change','live_survival_improvement_verified','checkpoint')}))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    main()
