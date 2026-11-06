"""Run one seed of source-only score validation or a fixed paper-protocol baseline."""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uniadet import cli,workflow


def build_configs(source,*,phase,seed,study_root,targets,source_data,score='baseline',holdout=('pcb1','pcb2')):
    train=copy.deepcopy(source['TRAIN_CONFIG']);test=copy.deepcopy(source['TEST_CONFIG'])
    train.update(data=source_data,split='test',seed=seed,device='cuda:0',epochs=15,
                 preserve_anomaly_mask=True,image_score_gaussian_sigma=None,shots=0)
    test.update(data=list(targets),shots=[0],epochs=[15],start_epoch=1,checkpoint=None,seed=seed,
                device='cuda:0',summary_name=f'test_summary_from_{Path(source_data).name}.csv',
                categories=None,exclude_categories=None,image_score_gaussian_sigma=None)
    if phase=='source-validation':
        if Path(source_data).name!='VisA' or targets!=[source_data]:
            raise ValueError('Source validation must train and validate on disjoint VisA categories')
        if not holdout or len(set(holdout))!=len(holdout):raise ValueError('Select unique source holdout categories')
        train.update(categories=None,exclude_categories=list(holdout))
        test.update(categories=list(holdout),exclude_categories=None)
    else:
        train.update(categories=None,exclude_categories=None)
        required=workflow.paper_auxiliary_groups(targets)
        if set(required)!={source_data}:raise ValueError('Benchmark auxiliary source does not match target protocol')
        if score=='raw_image_peak' and any(Path(t).name.lower()=='visa' for t in targets):
            raise ValueError('A rule selected on VisA holdouts cannot claim independent VisA generalization; use an independent target')
    backbone=train['backbone'];regime_dir=Path(study_root)/'baseline'/backbone/'0_shot'
    base=workflow.RunConfig(model=f'baseline-review-{phase}-{Path(source_data).name}-seed{seed}',train=train,test=test,
                            seed_output=str(regime_dir/f'seed_{seed}'))
    variants=['baseline','raw_image_peak'] if phase=='source-validation' or score=='raw_image_peak' else ['baseline']
    evaluations={}
    for variant in variants:
        current=copy.deepcopy(base)
        current.seed_output=str(Path(study_root)/variant/backbone/'0_shot'/f'seed_{seed}')
        if variant=='raw_image_peak':current.test['image_score_gaussian_sigma']=0.
        evaluations[variant]=current
    return base,evaluations


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--phase',choices=['source-validation','benchmark'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--source-data',required=True)
    p.add_argument('--target-data',required=True,help='Comma-separated target roots')
    p.add_argument('--score',choices=['baseline','raw_image_peak'],default='baseline')
    p.add_argument('--holdout',default='pcb1,pcb2')
    p.add_argument('--mode',choices=['train','test','train_test'],default='train_test')
    p.add_argument('--dry-run',action='store_true')
    return p


def configuration(options):
    source=runpy.run_path(str(options.config))
    return build_configs(source,phase=options.phase,seed=options.seed,study_root=options.study_root,
                         targets=options.target_data.split(','),source_data=options.source_data,
                         score=options.score,holdout=tuple(options.holdout.split(',')))


def main(argv=None):
    options=parser().parse_args(argv);base,evaluations=configuration(options)
    args=workflow.training_args(base)
    plans={v:dict(training=vars(args),test=c.test,
                  outputs=[str(workflow.output_path(c,target)/'epoch_15') for target in c.test['data']])
           for v,c in evaluations.items()}
    if options.dry_run:
        print(json.dumps(dict(mode=options.mode,phase=options.phase,checkpoint=str(Path(args.output)/'epoch_15.pt'),
                              variants=plans),indent=2));return
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','2','3'):
        raise RuntimeError('Map exactly one healthy physical GPU with CUDA_VISIBLE_DEVICES=0, 2, or 3')
    if args.device!='cuda:0' or not cli.torch.cuda.is_available():raise RuntimeError('Require healthy CUDA; CPU fallback is disabled')
    # Check all evaluation destinations before starting expensive training.
    if options.mode in ('test','train_test'):
        for plan in plans.values():
            for output in plan['outputs']:
                if Path(output).exists():raise ValueError(f'Output collision: {output}; use a new study ID')
    if options.mode in ('train','train_test'):
        cli.seed_all(args.seed);cli.train(args)
    if options.mode in ('test','train_test'):
        for config in evaluations.values():
            # args.output remains the single trained baseline checkpoint source.
            workflow.test_checkpoints(config,args,1)


if __name__=='__main__':main()
