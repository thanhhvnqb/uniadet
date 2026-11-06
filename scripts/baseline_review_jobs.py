"""Prepare immutable baseline-review jobs and aggregate completed evaluations."""
import argparse
import csv
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import pprint
import runpy
import shlex
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import reproduction_jobs as jobs
from scripts.run_baseline_review import build_configs
from uniadet import cli,workflow


def prepare(options):
    source=runpy.run_path(str(Path(options.config).resolve()));review=source['REVIEW_CONFIG']
    if options.variants!='baseline' or options.shots!=0:raise ValueError('Baseline review supports only --variants baseline --shots 0')
    if not options.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in options.run_id):raise ValueError('Invalid run ID')
    phase=review['phase'];score=review['score']
    if phase not in ('source-validation','benchmark') or score not in ('baseline','raw_image_peak'):raise ValueError('Invalid explicit review settings')
    requested=options.gpus.split(',')
    if not requested or len(requested)>3 or len(set(requested))!=len(requested) or set(requested)-{'0','2','3'}:raise ValueError('Use only distinct GPUs 0,2,3')
    if options.workers<0 or options.threads<1 or options.min_free_mib<1 or options.ram_per_job_gib<=0:raise ValueError('Invalid resource budgets')
    available=[];rejected=[]
    for gpu in requested:
        try:available.append(jobs.probe_gpu(gpu,options.min_free_mib))
        except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as error:
            rejected.append(dict(gpu=gpu,reason=str(error)))
            print(f'GPU {gpu} unavailable: {error}',file=sys.stderr)
    if not available:raise RuntimeError('No healthy available GPUs; CPU fallback disabled')
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    ram=int(memory['MemAvailable'].split()[0])*1024
    limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and limit.read_text().strip().isdigit() and current.exists():ram=min(ram,int(limit.read_text())-int(current.read_text()))
    slots=min(len(available),int(ram//(options.ram_per_job_gib*1024**3)))
    if slots<1:raise RuntimeError('Insufficient RAM budget')
    targets=options.target_data.split(',') if options.target_data else list(source['TEST_CONFIG']['data'])
    groups={'datasets/VisA':['datasets/VisA']} if phase=='source-validation' else workflow.paper_auxiliary_groups(targets)
    if phase=='source-validation' and targets!=['datasets/VisA']:raise ValueError('Source validation targets must be datasets/VisA')
    study=(ROOT/review['root']/options.run_id/phase).resolve()
    plan_dir=ROOT/'runs/baseline/launches'/options.run_id
    if study.exists() or plan_dir.exists():raise ValueError('Study/run ID already exists; preserve it and choose a new ID')
    records=[];manifests={}
    for auxiliary,target_group in groups.items():
        for data in [auxiliary,*target_group]:
            if not (Path(data)/'meta.json').is_file():raise ValueError(f'Missing metadata: {data}')
        if phase=='source-validation':
            metadata=json.loads((Path(auxiliary)/'meta.json').read_text())
            if not set(review['holdout'])<set(metadata['test']):raise ValueError('Holdout must be a proper subset of VisA categories')
        for index,seed in enumerate((10,20,30)):
            frozen={k:copy.deepcopy(source[k]) for k in ('TRAIN_CONFIG','TEST_CONFIG')}
            frozen['TRAIN_CONFIG'].update(workers=options.workers,device='cuda:0')
            frozen['TEST_CONFIG'].update(workers=options.workers,device='cuda:0',batch_size=options.test_batch_size)
            base,evaluations=build_configs(frozen,phase=phase,seed=seed,study_root=study,targets=target_group,
                                          source_data=auxiliary,score=score,holdout=review['holdout'])
            args=workflow.training_args(base);checkpoint=Path(args.output)/'epoch_15.pt'
            job_dir=Path(base.seed_output);record=job_dir/f'job_{Path(auxiliary).name}.json'
            snapshot=study/'configs'/f'seed{seed}_from_{Path(auxiliary).name}.py'
            common=[sys.executable,'-u',str(ROOT/'scripts/run_baseline_review.py'),'--config',str(snapshot),
                    '--phase',phase,'--seed',str(seed),'--study-root',str(study),'--source-data',auxiliary,
                    '--target-data',','.join(target_group),'--score',score,'--holdout',','.join(review['holdout'])]
            ev=[]
            for variant,config in evaluations.items():
                report_dir=Path(config.seed_output).parent
                manifests[str(report_dir)]=dict(variant=variant,backbone=args.backbone,regime='zero_shot',shots=0,
                    seeds=[10,20,30],phase=phase,run_id=options.run_id,checkpoint_epoch=15,training_split='test',evaluation_split='test',
                    holdout=review['holdout'],matched_baseline=str(study/'baseline'/args.backbone/'0_shot/manifest.json'),
                    state='planned',jobs=[],configs=[])
                for target in target_group:
                    directory=workflow.output_path(config,target)/'epoch_15'
                    ev.append(dict(variant=variant,dataset=Path(target).name,training_dataset=Path(auxiliary).name,
                        checkpoint=str(checkpoint),metrics=str(directory/'metrics.json'),evaluation_config=str(directory/'evaluation_config.json'),
                        inference_overrides={'image_score_gaussian_sigma':0.} if variant=='raw_image_peak' else {}))
            record_data=dict(id=f'baseline-review-{phase}-seed{seed}-{Path(auxiliary).name}',variant='baseline',backbone=args.backbone,
                regime='zero_shot',shots=0,seed=seed,gpu=available[index%len(available)]['physical_gpu'],training_dataset=Path(auxiliary).name,
                snapshot=str(snapshot),snapshot_source=frozen,record=str(record),log=str(job_dir/f'job_{Path(auxiliary).name}.log'),
                training=vars(args),test=base.test,checkpoint=str(checkpoint),metrics=ev[0]['metrics'],evaluation_config=ev[0]['evaluation_config'],
                evaluations=ev,commands=[common+['--mode',mode] for mode in ['train','test']],minimum_free_mib=options.min_free_mib,
                state='queued',created=jobs.now())
            records.append(record_data)
            print(f'GPU={record_data["gpu"]} seed={seed} input={auxiliary} -> {target_group}\ncheckpoint={checkpoint}\noutputs={[e["metrics"] for e in ev]}')
            print(json.dumps(dict(phase=phase,training=vars(args),test_overrides={v:c.test for v,c in evaluations.items()}),indent=2))
            for command in record_data['commands']:print(shlex.join(['CUDA_VISIBLE_DEVICES='+record_data['gpu'],*command]))
    print(f'{len(records)} training jobs, {sum(len(j["evaluations"]) for j in records)} evaluations; up to {slots} concurrent GPUs; {options.workers} workers + {options.threads} CPU threads/job')
    if not options.execute:return
    plan_dir.mkdir(parents=True);study.mkdir(parents=True)
    (plan_dir/'working-tree.patch').write_text(jobs.command_output(['git','diff','HEAD'])+'\n')
    code=plan_dir/'code';hashes={}
    for name in jobs.command_output(['git','ls-files','--cached','--others','--exclude-standard','--','*.py','*.sh','AGENTS.md','uv.lock']).splitlines():
        path=ROOT/name
        if not path.is_file() or name.startswith(('runs/','datasets/')):continue
        destination=code/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes());hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    environment=dict(commit=jobs.command_output(['git','rev-parse','HEAD']),working_tree=jobs.command_output(['git','status','--short']),
                     python=sys.executable,platform=platform.platform(),gpu=available,rejected_gpu=rejected,requested_gpu=requested,
                     source_hashes=hashes,workers=options.workers,threads=options.threads,
                     available_ram_bytes=ram,ram_per_job_gib=options.ram_per_job_gib,max_parallel=slots)
    for record in records:
        snapshot=Path(record['snapshot']);snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text('# Immutable baseline-review configuration\n'+'\n'.join(f'{k} = {pprint.pformat(v,sort_dicts=False)}' for k,v in record.pop('snapshot_source').items())+'\n');snapshot.chmod(0o444)
        record['snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest();Path(record['record']).parent.mkdir(parents=True,exist_ok=True);jobs.write_json(record['record'],record)
    for directory,manifest in manifests.items():
        selected=[r for r in records if any(str(Path(e['metrics']).parents[3])==directory for e in r['evaluations'])]
        manifest.update(jobs=[r['record'] for r in selected],worker_jobs=[r['record'] for r in selected],configs=[r['snapshot'] for r in selected],
                        environment=str(plan_dir/'environment.json'),code_snapshot=str(code),working_tree_diff=str(plan_dir/'working-tree.patch'))
        Path(directory).mkdir(parents=True,exist_ok=True)
        jobs.write_json(Path(directory)/'manifest.json',manifest)
    args=argparse.Namespace(**records[0]['training']);args.device='cpu';model=cli.build(args)
    environment['weight_source']=dict(backbone=args.backbone,weights=args.weights,dino_repo=args.dino_repo,
        hf_commit=getattr(getattr(getattr(model.encoder,'model',None),'config',None),'_commit_hash',None))
    environment['head_parameters']=sum(p.numel() for p in model.head.parameters());jobs.write_json(plan_dir/'environment.json',environment)
    jobs.write_json(plan_dir/'plan.json',dict(phase=phase,study=str(study),max_parallel=slots,jobs=[r['record'] for r in records],manifests=[str(Path(d)/'manifest.json') for d in manifests]))
    (plan_dir/'max_parallel').write_text(str(slots)+'\n')
    (plan_dir/'queue.tsv').write_text(''.join(f'{r["gpu"]}\t{r["record"]}\t{r["log"]}\n' for r in records))


def aggregate(plan_path):
    plan=json.loads(Path(plan_path).read_text());proxies=[]
    statuses={}
    exits=Path(plan_path).parent/'exits.tsv'
    if exits.exists():statuses=dict(line.rsplit('\t',1) for line in exits.read_text().splitlines())
    directory=Path(plan_path).parent/'aggregation';directory.mkdir(exist_ok=True)
    for record in plan['jobs']:
        job=json.loads(Path(record).read_text())
        if record in statuses and int(statuses[record])!=0:job.update(state='failed',returncode=int(statuses[record]))
        for index,evaluation in enumerate(job['evaluations']):
            proxy=dict(job,**evaluation);proxy.update(id=job['id']+':'+evaluation['variant']+':'+evaluation['dataset'],record=str(directory/f'{len(proxies)}.json'),evaluations=[evaluation])
            jobs.write_json(proxy['record'],proxy);proxies.append(proxy)
    for manifest_path in plan['manifests']:
        manifest=json.loads(Path(manifest_path).read_text());report_dir=str(Path(manifest_path).parent)
        selected=[j for j in proxies if str(Path(j['metrics']).parents[3])==report_dir]
        manifest['jobs']=[j['record'] for j in selected];jobs.write_json(manifest_path,manifest)
    proxy_plan=directory/'plan.json';jobs.write_json(proxy_plan,dict(jobs=[j['record'] for j in proxies],manifests=plan['manifests']))
    status=jobs.aggregate(proxy_plan)
    # Preserve the actual protocol; the shared reporter assumes cross-dataset training.
    for manifest_path in plan['manifests']:
        manifest=json.loads(Path(manifest_path).read_text());d=Path(manifest_path).parent
        description=('Source-only VisA category holdout: '+', '.join(manifest['holdout'])+' excluded from training and CAA; no target selection.' if plan['phase']=='source-validation'
                     else 'Fixed paper-protocol target evaluation; configuration fixed before target labels.')
        (d/'discussion.md').write_text(f'# {manifest["variant"]}: zero-shot baseline review\n\n## 1. Paper reference\n\nSee the versioned cache at reports/paper/uniadet_2601.05552v1/. Source-category validation is not a full paper benchmark and must not be numerically equated with it.\n\n## 2. Matched baseline\n\nBaseline manifest: '+manifest['matched_baseline']+'\n\n## 3. Experiment results\n\n'+description+'\n\nStatus: '+manifest['state']+'. See results.csv for categories, summary.csv for sample SD and paired deltas, and manifest.json for provenance. Both score variants reuse one trained head per source/seed; map smoothing and head capacity are identical.\n')
    write_review_report(plan)
    return status


def write_review_report(plan):
    from scripts.build_experiment_report import table,formatted,write_csv
    from uniadet.evaluation_metrics import METRIC_COLUMNS
    study=Path(plan['study']);all_results=[];all_summaries=[];manifests=[]
    for path in plan['manifests']:
        manifest=json.loads(Path(path).read_text());manifests.append(manifest);directory=Path(path).parent
        all_results.extend(csv.DictReader((directory/'results.csv').open()))
        all_summaries.extend(csv.DictReader((directory/'summary.csv').open()))
    write_csv(study/'family_results.csv',all_results);write_csv(study/'family_summary.csv',all_summaries)
    cached=ROOT/'reports/paper/uniadet_2601.05552v1/reference.json'
    paper=json.loads(cached.read_text())['datasets'] if cached.exists() else []
    rows=[]
    for dataset in sorted({r['dataset'] for r in all_summaries}):
        reference=next((r for r in paper if r['backbone']==manifests[0]['backbone'] and r['dataset']==dataset and r['shots']==0),None)
        common=dict(dataset=dataset,backbone=manifests[0]['backbone'],shots=0)
        p=dict(common,kind='paper_reported',variant='paper',completed_seeds='',status='external_context_only' if plan['phase']=='source-validation' else 'reported' if reference else 'unavailable')
        for _,_,metric in METRIC_COLUMNS:
            value=(reference or {}).get('metrics',{}).get(metric,{})
            p[metric]=value.get('reported','—')
        rows.append(p)
        for manifest in manifests:
            variant=manifest['variant'];selected=[r for r in all_summaries if r['dataset']==dataset and r['variant']==variant]
            if not selected:continue
            row=dict(common,kind='reproduced_baseline' if variant=='baseline' else 'experiment',variant=variant,
                     completed_seeds=selected[0]['completed_seeds'],status=manifest['state'])
            for _,_,metric in METRIC_COLUMNS:
                value=next(r for r in selected if r['metric']==metric)
                row[metric]=formatted(value['mean_percent'],value['sample_std_percent'])
            rows.append(row)
    write_csv(study/'comparison.csv',rows)
    text='# Baseline reproduction review\n\n## 1. Paper-reported reference\n\n'
    if plan['phase']=='source-validation':text+='Full-dataset paper references below are context only: this study validates unseen source categories and is not the paper cross-dataset benchmark. Do not compute a paper-reproduction gain from these different protocols.\n\n'
    text+=table(['Dataset','I-AUROC','P-AUPR','Status'],[(r['dataset'],r['I-AUROC'],r['P-AUPR'],r['status']) for r in rows if r['variant']=='paper'])+'\n'
    text+='## 2. Reproduced matched baseline\n\n'+table(['Dataset','Seeds','I-AUROC','P-AUPR','Status'],[(r['dataset'],r['completed_seeds']+'/3',r['I-AUROC'],r['P-AUPR'],r['status']) for r in rows if r['variant']=='baseline'])+'\n'
    text+='## 3. Other experiment results\n\n'+table(['Dataset','Variant','Seeds','I-AUROC','P-AUPR','Status'],[(r['dataset'],r['variant'],r['completed_seeds']+'/3',r['I-AUROC'],r['P-AUPR'],r['status']) for r in rows if r['variant'] not in ('paper','baseline')])+'\n'
    decision='not_applicable'
    if plan['phase']=='source-validation':
        candidate=next(r for r in all_summaries if r['variant']=='raw_image_peak' and r['metric']=='I-AUROC')
        base=next(r for r in all_summaries if r['variant']=='baseline' and r['metric']=='I-AUROC')
        pixel=next(r for r in all_summaries if r['variant']=='raw_image_peak' and r['metric']=='P-AUPR')
        complete=all(r['completed_seeds']=='3' and r['valid_seeds']=='3' for r in (candidate,base,pixel))
        if not complete:decision='incomplete_do_not_select'
        elif float(candidate['paired_delta_pp'])>0 and int(candidate['seeds_improved'])>=2 and abs(float(pixel['paired_delta_pp']))<1e-9:decision='raw_image_peak_eligible_for_independent_confirmation'
        else:decision='retain_baseline'
        text+='Predeclared source-only decision: **'+decision+'**. Require three valid seeds, positive paired mean I-AUROC, at least two seeds improved, and unchanged P-AUPR. Freeze the chosen rule before target evaluation. The two-category holdout is limited evidence and does not establish significance or paper-score reproduction.\n\n'
    text+='Both score rules reuse the same trained head for each source/seed. Map smoothing, parameter count, augmentation and losses are fixed. Zero-shot uses no target references. Full seven-metric results, category changes, paired deltas, valid category counts and per-seed sources are in family_results.csv/family_summary.csv and variant manifests; no missing result is zero-filled.\n\nGPU/weight/code provenance, commands, logs, wall time and worker outcomes are linked from manifests. The evaluator logs loading/inference and metrics separately. Peak VRAM is not instrumented by this runner; no speed or memory-performance gain is claimed.\n'
    (study/'comparison.md').write_text(text.rstrip()+'\n')
    jobs.write_json(study/'selection.json',dict(phase=plan['phase'],decision=decision,source_only=plan['phase']=='source-validation'))
    for path in plan['manifests']:
        directory=Path(path).parent
        with (directory/'discussion.md').open('a') as f:f.write('\nCoordinator comparison and predeclared decision: '+str(study/'comparison.md')+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name,default in [('config','configs/reproduction_source_validation.py'),('variants','baseline'),('gpus','0,2,3'),('run-id',None),('target-data',None)]:prep.add_argument('--'+name,default=default,required=name=='run-id')
    for name,default in [('shots',0),('workers',2),('threads',2),('min-free-mib',9500),('ram-per-job-gib',24),('test-batch-size',16)]:prep.add_argument('--'+name,type=int,default=default)
    prep.add_argument('--execute',action='store_true');prep.add_argument('--dry-run',action='store_true')
    for name,arg in [('job','record'),('aggregate','plan')]:sub.add_parser(name).add_argument(arg)
    o=p.parse_args(argv)
    if o.mode=='prepare':prepare(o)
    elif o.mode=='job':jobs.run_job(o.record)
    else:return aggregate(o.plan)


if __name__=='__main__':sys.exit(main() or 0)
