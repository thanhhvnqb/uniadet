import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from config import TRAIN_CONFIG,TEST_CONFIG
from scripts import baseline_review_jobs as coordinator
from scripts.run_baseline_review import build_configs
from uniadet import workflow


def test_source_configs_share_checkpoint_and_keep_source_categories_disjoint(tmp_path):
    source=dict(TRAIN_CONFIG=copy.deepcopy(TRAIN_CONFIG),TEST_CONFIG=copy.deepcopy(TEST_CONFIG))
    original=copy.deepcopy(source)
    base,variants=build_configs(source,phase='source-validation',seed=20,study_root=tmp_path,
                                targets=['datasets/VisA'],source_data='datasets/VisA')
    args=workflow.training_args(base)
    assert args.exclude_categories==['pcb1','pcb2'] and args.categories is None
    assert args.epochs==15 and args.seed==20
    assert source==original
    assert set(variants)=={'baseline','raw_image_peak'}
    assert all(c.test['categories']==['pcb1','pcb2'] for c in variants.values())
    assert variants['baseline'].test['image_score_gaussian_sigma'] is None
    assert variants['raw_image_peak'].test['image_score_gaussian_sigma']==0.
    assert variants['raw_image_peak'].test['gaussian_sigma'] is None
    assert variants['baseline'].seed_output!=variants['raw_image_peak'].seed_output
    # The shared workflow uses args.output as checkpoint source even for a different evaluation output.
    assert 'baseline/dinov3/0_shot/seed_20/train_VisA' in args.output


def test_benchmark_routes_sources_and_does_not_sweep_target_scores(tmp_path):
    source=dict(TRAIN_CONFIG=TRAIN_CONFIG,TEST_CONFIG=TEST_CONFIG)
    base,variants=build_configs(source,phase='benchmark',seed=10,study_root=tmp_path,
                               targets=['datasets/Real-IAD'],source_data='datasets/VisA')
    assert list(variants)==['baseline']
    assert base.train['exclude_categories'] is None and base.test['categories'] is None
    with pytest.raises(ValueError,match='auxiliary source'):
        build_configs(source,phase='benchmark',seed=10,study_root=tmp_path,
                      targets=['datasets/VisA'],source_data='datasets/VisA')
    with pytest.raises(ValueError,match='independent VisA generalization'):
        build_configs(source,phase='benchmark',seed=10,study_root=tmp_path,
                      targets=['datasets/VisA'],source_data='datasets/MVTecAD',score='raw_image_peak')


@pytest.mark.parametrize('failed',[False,True])
def test_review_plan_and_aggregation_do_not_mix_score_variants(tmp_path,monkeypatch,failed):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(coordinator,'ROOT',tmp_path)
    (tmp_path/'datasets/VisA').mkdir(parents=True)
    (tmp_path/'datasets/VisA/meta.json').write_text(json.dumps(dict(test=dict(pcb1=[],pcb2=[],candle=[]))))
    config=tmp_path/'config.py'
    config.write_text('\n'.join(f'{k} = {v!r}' for k,v in dict(TRAIN_CONFIG=TRAIN_CONFIG,TEST_CONFIG=TEST_CONFIG,
        REVIEW_CONFIG=dict(phase='source-validation',score='baseline',holdout=['pcb1','pcb2'],root='runs/baseline/reproduction_review')).items()))
    monkeypatch.setattr(coordinator.jobs,'probe_gpu',lambda gpu,*_:dict(physical_gpu=gpu))
    monkeypatch.setattr(coordinator.jobs,'command_output',lambda *_:'')
    monkeypatch.setattr(coordinator.cli,'build',lambda *_:SimpleNamespace(encoder=SimpleNamespace(),head=torch.nn.Linear(4,2)))
    options=argparse.Namespace(config=str(config),variants='baseline',shots=0,run_id='unit',gpus='0,2,3',workers=0,
        threads=1,min_free_mib=1,ram_per_job_gib=1,target_data='datasets/VisA',test_batch_size=4,execute=True)
    coordinator.prepare(options)
    plan_path=tmp_path/'runs/baseline/launches/unit/plan.json';plan=json.loads(plan_path.read_text())
    assert len(plan['jobs'])==3 and len(plan['manifests'])==2
    for i,path in enumerate(plan['jobs']):
        j=json.loads(Path(path).read_text());assert len(j['evaluations'])==2
        assert not Path(j['snapshot']).stat().st_mode&0o222
        assert j['gpu']==['0','2','3'][i]
        j['state']='failed' if failed and i==2 else 'complete'
        Path(path).write_text(json.dumps(j))
        for e in j['evaluations']:
            value=.5+i*.01+(.02 if e['variant']=='raw_image_peak' else 0)
            category=dict(image=dict(auroc=value,aupr=value,f1max=value),pixel=dict(auroc=value,aupr=value,f1max=value,aupro=value))
            p=Path(e['metrics']);p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(pcb1=category,pcb2=category,average=category)))
    assert coordinator.aggregate(plan_path)==int(failed)
    for path in plan['manifests']:
        manifest=json.loads(Path(path).read_text());assert len(manifest['jobs'])==3
        assert manifest['state']==('incomplete' if failed else 'complete')
        rows=list(csv.DictReader((Path(path).parent/'summary.csv').open()))
        selected=next(r for r in rows if r['metric']=='P-AUPR')
        assert selected['completed_seeds']==('2' if failed else '3')
        assert float(selected['paired_delta_pp'])==pytest.approx(2. if manifest['variant']=='raw_image_peak' else 0.)
