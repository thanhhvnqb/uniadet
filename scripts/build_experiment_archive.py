"""Rebuild offline reporting tables; no training, evaluation, or paper downloads."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = {'I-AUROC': ('image', 'auroc'), 'I-AUPR': ('image', 'aupr'),
           'I-F1max': ('image', 'f1max'), 'P-AUROC': ('pixel', 'auroc'),
           'P-AUPR': ('pixel', 'aupr'), 'P-F1max': ('pixel', 'f1max'),
           'P-AUPRO': ('pixel', 'aupro')}
SCORE = ROOT / 'runs/baseline/reproduction_review/source-score-20261106T043532Z/source-validation'
AUDIT = ROOT / 'runs/baseline/dinov3/0_shot/audit_20261106'


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def verify(rows):
    sources = sorted({r['source'] for r in rows if r['status'] == 'complete'})
    for source in sources:
        path = Path(source)
        raw = json.loads(path.read_text())
        saved = json.loads((path.parent / 'evaluation_config.json').read_text())
        config = saved['evaluation']
        assert config['backbone'] == 'dinov3' and config['shots'] == 0
        assert saved['checkpoint_epoch'] == 15
        categories = set(raw) - {'average', 'macro_mean'}
        selected = [r for r in rows if r['source'] == source and r['status'] == 'complete']
        assert {r['category'] for r in selected} - {'average', 'macro_mean'} == categories
        for row in selected:
            assert int(row['seed']) == config['seed']
            for metric, (group, key) in METRICS.items():
                value = (statistics.mean(raw[c][group][key] for c in categories)
                         if row['category'] in {'average', 'macro_mean'}
                         else raw[row['category']][group][key])
                assert abs(float(row[metric]) - 100 * value) < 1e-8, (source, metric)
    return len(sources)


def verify_summary(rows, summaries):
    for row in summaries:
        selected = [r for r in rows if r['status'] == 'complete'
                    and r['variant'] == row['variant'] and r['dataset'] == row['dataset']
                    and r['category'] == 'macro_mean']
        assert len(selected) == int(row['completed_seeds'])
        if not selected:
            assert not row['mean_percent']
            continue
        metric = row['metric']
        values = [float(r[metric]) for r in selected]
        assert len({r['seed'] for r in selected}) == len(selected)
        assert abs(statistics.mean(values) - float(row['mean_percent'])) < 1e-8
        assert abs(statistics.stdev(values) - float(row['sample_std_percent'])) < 1e-8
        baseline = {r['seed']: r for r in rows if r['status'] == 'complete'
                    and r['variant'] == 'baseline' and r['dataset'] == row['dataset']
                    and r['category'] == 'macro_mean'}
        deltas = [float(r[metric])-float(baseline[r['seed']][metric]) for r in selected]
        assert abs(statistics.mean(deltas) - float(row['paired_delta_pp'])) < 1e-8
        assert sum(d > 0 for d in deltas) == int(row['seeds_improved'])


def main():
    output = ROOT / 'reports/feature_ablation'
    gap = ROOT / 'reports/reproduction_gap'
    output.mkdir(exist_ok=True)
    gap.mkdir(exist_ok=True)
    paths = [AUDIT / 'family_results.csv', AUDIT / 'family_summary.csv',
             SCORE / 'family_results.csv', SCORE / 'family_summary.csv',
             ROOT / 'reports/context/experimental_results.csv',
             AUDIT / 'summary.csv', ROOT / 'reports/experiment_comparison.csv',
             ROOT / 'reports/experiment_category_comparison.csv',
             ROOT / 'reports/paper/uniadet_2601.05552v1/reference.json']
    family = read(paths[0])
    holdout = read(paths[2])
    count = verify(family)
    holdout_count = verify(holdout)
    verify_summary(family, read(paths[1]))
    verify_summary(holdout, read(paths[3]))
    for source in {r['source'] for r in holdout}:
        saved = json.loads((Path(source).parent / 'evaluation_config.json').read_text())
        assert set(saved['training']['exclude_categories']) == {'pcb1', 'pcb2'}
        assert set(saved['evaluation']['categories']) == {'pcb1', 'pcb2'}
    for name, rows in [('results.csv', family), ('summary.csv', read(paths[1])),
                       ('source_validation_results.csv', holdout),
                       ('source_validation_summary.csv', read(paths[3])),
                       ('context_results_archived.csv', read(paths[4])),
                       ('score_diagnostics.csv', read(paths[5]))]:
        write(output / name, rows)
    inventory = []
    for protocol, rows in [('cross_dataset', family), ('source_holdout', holdout)]:
        for source in sorted({r['source'] for r in rows if r['status'] == 'complete'}):
            path = Path(source)
            config_path = path.parent / 'evaluation_config.json'
            saved = json.loads(config_path.read_text())
            config = saved['evaluation']
            inventory.append(dict(protocol=protocol, backbone=config['backbone'],
                                  shots=config['shots'], seed=config['seed'],
                                  training_dataset=Path(saved['training']['data']).name,
                                  target_dataset=Path(config['data']).name,
                                  patch_features=config['patch_features'], checkpoint_epoch=15,
                                  checkpoint=config['checkpoint'],
                                  checkpoint_exists=Path(config['checkpoint']).is_file(),
                                  metrics=source, evaluation_config=str(config_path),
                                  metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
    write(output / 'artifact_inventory.csv', inventory)
    # Paper-first tables retain all seven metrics and category provenance.
    for name in ['experiment_comparison.csv', 'experiment_category_comparison.csv']:
        write(output / name, read(ROOT / 'reports' / name))
    reference = json.loads(paths[-1].read_text())
    gaps = []
    for kind, category in [('datasets', 'macro_mean'), ('categories', None)]:
        for paper in reference[kind]:
            if paper['backbone'] != 'dinov3' or paper['shots'] != 0:
                continue
            cat = category or paper['category']
            local = [r for r in family if r['variant'] == 'baseline'
                     and r['dataset'] == paper['dataset'] and r['category'] == cat
                     and r['status'] == 'complete']
            if not local:
                continue
            for metric in METRICS:
                external = paper['metrics'].get(metric, {})
                if external.get('value_percent') is None:
                    continue
                values = [float(r[metric]) for r in local]
                assert len(values) == 3
                mean = statistics.mean(values)
                gaps.append(dict(dataset=paper['dataset'], category=cat, backbone='dinov3',
                                 shots=0, metric=metric, paper_percent=external['value_percent'],
                                 paper_reported=external['reported'], source_table=paper['source_table'],
                                 local_mean_percent=mean, local_sample_sd=statistics.stdev(values),
                                 local_minus_paper_pp=mean-external['value_percent'],
                                 seeds='10;20;30', checkpoint_epoch=15,
                                 comparison='unpaired_external',
                                 source=';'.join(r['source'] for r in local)))
    write(gap / 'paper_gaps.csv', gaps)
    context = read(paths[4])
    missing = [p for r in context for p in r['source_metrics_files'].split(';') if not (ROOT/p).is_file()]
    provenance = dict(as_of='2026-11-06', operation='offline_reporting_only',
                      cross_dataset_evaluations_verified=count,
                      holdout_evaluations_verified=holdout_count,
                      summary_mean_sample_sd_paired_deltas='verified against per-seed rows',
                      context_archived_summary_rows=len(context),
                      context_missing_raw_paths=len(set(missing)),
                      context_validation='archived summaries only; raw data unavailable here',
                      inputs=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k:v for k,v in provenance.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
