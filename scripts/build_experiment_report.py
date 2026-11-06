"""Build paper → reproduced baseline → experiments reports entirely from local files."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
METRICS=['I-AUROC','I-AUPR','I-F1max','P-AUROC','P-AUPR','P-F1max','P-AUPRO']
VARIANTS=['baseline','cluster','components','directional','knn','residual','spatial']

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'

def formatted(value,std=None):
    if value is None or value=='':return '—'
    return f'{float(value):.2f}'+(f' ± {float(std):.2f}' if std is not None and std!='' else '')

def format_row(row,metric):
    if row['variant']=='paper':return row.get(metric+'_reported') or '—'
    return formatted(row[metric],row[metric+'_std'])

def comparison_rows(reference,local,backbone,shots):
    selected=[r for r in local if int(r['shots'])==shots]
    datasets=sorted({r['dataset'] for r in selected})
    rows=[]
    for dataset in datasets:
        categories=['macro_mean']+sorted({r['category'] for r in selected if r['dataset']==dataset and r['category']!='macro_mean'})
        for category in categories:
            paper=next((r for r in reference['datasets' if category=='macro_mean' else 'categories']
                        if r['backbone']==backbone and r['dataset']==dataset and r['shots']==shots and r['category']==category),None)
            for order,variant in enumerate(['paper',*VARIANTS]):
                seeds=sorted({int(r['seed']) for r in selected if r['dataset']==dataset and r['category']==category and r['variant']==variant and r['status']=='complete'})
                row=dict(order=order,kind='paper_reported' if variant=='paper' else 'reproduced_baseline' if variant=='baseline' else 'experiment',variant=variant,
                         backbone=backbone,regime='zero_shot' if shots==0 else 'few_shot',shots=shots,dataset=dataset,category=category,
                         completed_seeds=None if variant=='paper' else len(seeds),checkpoint_epoch=None if variant=='paper' else 15,
                         status=('reported' if paper else 'reference_unavailable') if variant=='paper' else 'complete' if len(seeds)==3 else 'incomplete_or_not_run')
                for m in METRICS:
                    values=[float(r[m]) for r in selected if r['dataset']==dataset and r['category']==category and r['variant']==variant and r['status']=='complete' and r[m]!='']
                    assert len(values)<=3
                    p=(paper or {}).get('metrics',{}).get(m,{})
                    row[m]=p.get('value_percent') if variant=='paper' else statistics.mean(values) if values else None
                    row[m+'_std']=p.get('std_percent') if variant=='paper' else statistics.stdev(values) if len(values)>1 else None
                    row[m+'_reported']=p.get('reported') if variant=='paper' else None
                    row[m+'_valid_seeds']=None if variant=='paper' else len(values)
                row['source']=paper['source_url'] if variant=='paper' and paper else ';'.join(dict.fromkeys(r['source'] for r in selected if r['dataset']==dataset and r['category']==category and r['variant']==variant))
                rows.append(row)
    return rows

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'reports/paper/uniadet_2601.05552v1/reference.json')
    parser.add_argument('--results',type=Path,default=ROOT/'runs/baseline/dinov3/0_shot/audit_20261106/family_results.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'reports')
    parser.add_argument('--backbone',default='dinov3',choices=['clip','dinov2','dinov3'])
    parser.add_argument('--shots',type=int,default=0,choices=[0,1,2,4])
    args=parser.parse_args(argv)
    reference=json.loads(args.reference.read_text());local=list(csv.DictReader(args.results.open()))
    sources={r['source'] for r in local if r['status']=='complete' and int(r['shots'])==args.shots}
    for source in sources:
        config_path=Path(source).parent/'evaluation_config.json'
        if not config_path.is_file():
            raise ValueError(f'Cannot verify backbone provenance: {config_path}')
        saved=json.loads(config_path.read_text());config=saved['evaluation']
        if config['backbone']!=args.backbone:
            raise ValueError(f'Requested {args.backbone}, but {config_path} records {config["backbone"]}')
        source_rows=[r for r in local if r['source']==source and r['status']=='complete' and int(r['shots'])==args.shots]
        if (config['shots']!=args.shots or saved['checkpoint_epoch']!=15
                or any(int(r['seed'])!=config['seed'] or r['dataset']!=Path(config['data']).name for r in source_rows)):
            raise ValueError(f'Shot/seed/dataset/epoch provenance mismatch: {config_path}')
    rows=comparison_rows(reference,local,args.backbone,args.shots);args.output.mkdir(parents=True,exist_ok=True)
    assert rows,'No local results for requested shots'
    # Local provenance must establish backbone: the source CSV is explicitly selected by the caller.
    write_csv(args.output/'experiment_comparison.csv',[r for r in rows if r['category']=='macro_mean'])
    write_csv(args.output/'experiment_category_comparison.csv',[r for r in rows if r['category']!='macro_mean'])
    cache_link=os.path.relpath(args.reference.parent/'README.md',args.output)
    header=f'# Experiment comparison: {args.backbone}, shots={args.shots}\n\n'
    header+='Order throughout: paper-reported values → reproduced baseline → other experiments. Percentages; local ± is sample SD over seeds 10/20/30. Paper ± is retained as published, with no assumed seed pairing. Missing values remain unavailable.\n\n'
    report=header+'## 1. Paper-reported results\n\n'
    report+=table(['Dataset',*METRICS,'Table'],[(r['dataset'],*[r['metrics'].get(m,{}).get('reported','—') for m in METRICS],r['source_table']) for r in reference['datasets'] if r['backbone']==args.backbone and r['shots']==args.shots])
    report+=f'\nSource: [versioned paper cache]({cache_link}); it preserves all three backbones, published standard deviations, source tables, and inconsistencies.\n\n'
    report+='## 2. Locally reproduced matched baseline\n\n'
    baseline=[r for r in rows if r['variant']=='baseline' and r['category']=='macro_mean']
    report+=table(['Dataset','Seeds',*METRICS],[(r['dataset'],str(r['completed_seeds'])+'/3',*[formatted(r[m],r[m+'_std']) for m in METRICS]) for r in baseline])
    report+=f'\nFrozen {args.backbone}, labeled auxiliary test-split supervision, epoch 15, seeds 10/20/30. VisA → MVTecAD and MVTecAD → VisA are separate transfer protocols. Unavailable dataset baselines are not claimed. Baseline settings and per-seed artifact paths are preserved in the linked source results; diagnostic scoring alternatives are not substituted for baseline measurements.\n\n'
    report+='## 3. Other experiment results\n\n'
    for dataset in sorted({r['dataset'] for r in rows}):
        report+=f'### {dataset}\n\n'
        grouped=[r for r in rows if r['dataset']==dataset and r['category']=='macro_mean']
        report+=table(['Result','Seeds','P-AUPR % ± SD','I-AUROC % ± SD','Status'],[(r['variant'],'—' if r['variant']=='paper' else str(r['completed_seeds'])+'/3',format_row(r,'P-AUPR'),format_row(r,'I-AUROC'),r['status']) for r in grouped])+'\n'
    report+='All variants use the matched shot regime. Spatial remains unrun; residual includes the recovered VisA evaluations. Compare experiment gains with the local baseline; paper gaps are unpaired external comparisons. These rows do not establish statistical significance or eliminate trainable-capacity confounds.\n\n'
    report+='[Category comparisons](experiment_categories.md) · [Dataset comparison CSV](experiment_comparison.csv) · [Category comparison CSV](experiment_category_comparison.csv)\n\n'
    report+='Local source: `'+os.path.relpath(args.results,args.output)+'`. Rebuild offline with `python scripts/build_experiment_report.py`; no paper parsing, download, training, or evaluation is performed.\n'
    (args.output/'experiment_comparison.md').write_text(report)
    details=header+'All seven metrics are retained. Every category group begins with the matching external paper row, then the local baseline, then variants.\n\n'
    for dataset in sorted({r['dataset'] for r in rows}):
        details+=f'## {dataset}\n\n'
        for category in sorted({r['category'] for r in rows if r['dataset']==dataset and r['category']!='macro_mean'}):
            details+=f'### {category}\n\n'+table(['Result','Seeds',*METRICS],[(r['variant'],'—' if r['variant']=='paper' else str(r['completed_seeds'])+'/3',*[format_row(r,m) for m in METRICS]) for r in rows if r['dataset']==dataset and r['category']==category])+'\n'
    (args.output/'experiment_categories.md').write_text(details.rstrip()+'\n')
    print(f'Wrote paper-first comparisons: {len(rows)} dataset/category rows, using cached paper data only')

if __name__=='__main__':main()
