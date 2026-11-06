import json
from pathlib import Path

import pytest

from scripts.build_experiment_report import comparison_rows, main
from scripts.cache_paper_results import measurement

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def reference():
    return json.loads((ROOT/'reports/paper/uniadet_2601.05552v1/reference.json').read_text())


def test_paper_reference_coverage_and_known_results(reference):
    assert len(reference['datasets'])==96
    assert len(reference['categories'])==876
    assert len({r['dataset'] for r in reference['datasets']})==14
    for dataset,auroc,aupr in [('MVTecAD',94.0,52.7),('VisA',91.9,32.5),('Real-IAD',81.2,41.6)]:
        row=next(r for r in reference['datasets'] if r['backbone']=='dinov3' and r['dataset']==dataset and r['shots']==0)
        assert row['metrics']['I-AUROC']['value_percent']==auroc
        assert row['metrics']['P-AUPR']['value_percent']==aupr
        assert row['metrics']['I-AUROC']['std_percent'] is None
    medical=next(r for r in reference['datasets'] if r['dataset']=='ISIC' and r['backbone']=='dinov3')
    assert 'I-AUROC' not in medical['metrics']
    assert medical['shots']==0
    assert not any(r['category']=='macro_mean' for r in reference['categories'])


def test_reported_inconsistencies_and_zero_std_are_preserved(reference):
    discrepancy=next(r for r in reference['discrepancies'] if r['backbone']=='dinov2' and r['shots']==2 and r['metric']=='I-AUROC')
    assert discrepancy['main_reported']=='89.0±0.7'
    assert discrepancy['detail_reported']=='89.8±0.4'
    assert len(reference['macro_discrepancies'])==5
    assert measurement('99.0±0.0')['std_percent']==0
    assert measurement('99.0')['std_percent'] is None
    with pytest.raises(ValueError):measurement('missing')


def test_comparison_order_and_local_sample_std(reference):
    local=[]
    for variant,delta in [('baseline',0),('knn',1)]:
        for seed,value in [(10,80),(20,82),(30,84)]:
            for category in ['bottle','macro_mean']:
                local.append(dict(variant=variant,seed=str(seed),shots='0',dataset='MVTecAD',category=category,
                    status='complete',source=f'{variant}/{seed}/metrics.json',**{m:str(value+delta) for m in ['I-AUROC','I-AUPR','I-F1max','P-AUROC','P-AUPR','P-F1max','P-AUPRO']}))
    rows=comparison_rows(reference,local,'dinov3',0)
    for category in ['bottle','macro_mean']:
        group=[r for r in rows if r['category']==category]
        assert [r['variant'] for r in group[:2]]==['paper','baseline']
        assert group[0]['completed_seeds'] is None
        assert group[1]['I-AUROC']==82
        assert group[1]['I-AUROC_std']==2
        assert group[1]['completed_seeds']==3
        missing=next(r for r in group if r['variant']=='spatial')
        assert missing['I-AUROC'] is None and missing['completed_seeds']==0
    # Never borrow a reference for a different backbone/shot/category.
    missing=next(r for r in comparison_rows(reference,[dict(local[0],category='unknown')],'dinov3',0)
                 if r['category']=='unknown' and r['variant']=='paper')
    assert missing['status']=='reference_unavailable' and missing['I-AUROC'] is None


@pytest.mark.parametrize('mismatch',['backbone','shots'])
def test_report_rejects_mislabeled_local_provenance(tmp_path,mismatch):
    import csv
    source=tmp_path/'metrics.json';source.write_text('{}')
    (tmp_path/'evaluation_config.json').write_text(json.dumps(dict(checkpoint_epoch=15,
        evaluation=dict(backbone='dinov3',shots=1 if mismatch=='shots' else 0,seed=10,data='datasets/MVTecAD'))))
    results=tmp_path/'results.csv'
    with results.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['status','shots','source','seed','dataset'])
        writer.writeheader();writer.writerow(dict(status='complete',shots=0,source=str(source),seed=10,dataset='MVTecAD'))
    with pytest.raises(ValueError,match='records dinov3|provenance mismatch'):
        main(['--results',str(results),'--output',str(tmp_path/'reports'),
              '--backbone','clip' if mismatch=='backbone' else 'dinov3'])
    assert not (tmp_path/'reports').exists()
