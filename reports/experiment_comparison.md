# Experiment comparison: dinov3, shots=0

Order throughout: paper-reported values → reproduced baseline → other experiments. Percentages; local ± is sample SD over seeds 10/20/30. Paper ± is retained as published, with no assumed seed pairing. Missing values remain unavailable.

## 1. Paper-reported results

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 94.0 | 97.1 | 94.0 | 93.2 | 52.7 | 51.1 | 90.7 | 14 |
| VisA | 91.9 | 92.8 | 88.1 | 96.3 | 32.5 | 36.0 | 91.7 | 17 |
| BTAD | 94.7 | 97.0 | 93.1 | 97.8 | 58.9 | 57.8 | 87.8 | 20 |
| DTD | 97.1 | 99.0 | 96.5 | 99.1 | 75.9 | 69.9 | 96.3 | 23 |
| KSDD | 91.0 | 64.5 | 67.8 | 99.0 | 22.6 | 32.6 | 92.0 | 26 |
| HeadCT | 95.7 | 94.8 | 90.1 | — | — | — | — | 27 |
| BrainMRI | 91.9 | 95.0 | 88.3 | — | — | — | — | 27 |
| Br35H | 92.3 | 93.8 | 85.3 | — | — | — | — | 27 |
| ISIC | — | — | — | 95.9 | 90.6 | 83.6 | 92.6 | 28 |
| ColonDB | — | — | — | 86.7 | 33.3 | 43.2 | 80.6 | 28 |
| ClinicDB | — | — | — | 91.5 | 57.5 | 59.5 | 83.4 | 28 |
| Endo | — | — | — | 92.8 | 68.5 | 66.1 | 83.3 | 28 |
| Kvasir | — | — | — | 91.7 | 65.9 | 65.8 | 54.9 | 28 |
| Real-IAD | 81.2 | 79.3 | 72.8 | 97.5 | 41.6 | 43.9 | 91.9 | 33 |

Source: [versioned paper cache](paper/uniadet_2601.05552v1/README.md); it preserves all three backbones, published standard deviations, source tables, and inconsistencies.

## 2. Locally reproduced matched baseline

| Dataset | Seeds | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 3/3 | 92.47 ± 0.27 | 96.43 ± 0.22 | 92.69 ± 0.21 | 91.95 ± 0.11 | 51.61 ± 0.14 | 50.80 ± 0.12 | 89.99 ± 0.13 |
| VisA | 3/3 | 88.91 ± 0.53 | 90.83 ± 0.50 | 85.62 ± 0.09 | 96.40 ± 0.07 | 33.42 ± 0.42 | 37.08 ± 0.30 | 90.50 ± 0.07 |

Frozen dinov3, labeled auxiliary test-split supervision, epoch 15, seeds 10/20/30. VisA → MVTecAD and MVTecAD → VisA are separate transfer protocols. Unavailable dataset baselines are not claimed. Baseline settings and per-seed artifact paths are preserved in the linked source results; diagnostic scoring alternatives are not substituted for baseline measurements.

## 3. Other experiment results

### MVTecAD

| Result | Seeds | P-AUPR % ± SD | I-AUROC % ± SD | Status |
|---|---|---|---|---|
| paper | — | 52.7 | 94.0 | reported |
| baseline | 3/3 | 51.61 ± 0.14 | 92.47 ± 0.27 | complete |
| cluster | 3/3 | 49.70 ± 0.78 | 92.07 ± 0.55 | complete |
| components | 3/3 | 51.88 ± 0.51 | 92.20 ± 0.17 | complete |
| directional | 3/3 | 51.38 ± 0.24 | 92.51 ± 0.15 | complete |
| knn | 3/3 | 51.22 ± 0.75 | 92.17 ± 0.21 | complete |
| residual | 3/3 | 36.47 ± 1.28 | 89.73 ± 0.86 | complete |
| spatial | 0/3 | — | — | incomplete_or_not_run |

### VisA

| Result | Seeds | P-AUPR % ± SD | I-AUROC % ± SD | Status |
|---|---|---|---|---|
| paper | — | 32.5 | 91.9 | reported |
| baseline | 3/3 | 33.42 ± 0.42 | 88.91 ± 0.53 | complete |
| cluster | 3/3 | 33.44 ± 0.35 | 89.22 ± 0.02 | complete |
| components | 3/3 | 32.95 ± 0.30 | 89.03 ± 0.08 | complete |
| directional | 3/3 | 33.27 ± 0.34 | 89.20 ± 0.15 | complete |
| knn | 3/3 | 32.92 ± 0.25 | 89.24 ± 0.10 | complete |
| residual | 3/3 | 25.60 ± 0.45 | 87.98 ± 0.77 | complete |
| spatial | 0/3 | — | — | incomplete_or_not_run |

All variants use the matched shot regime. Spatial remains unrun; residual includes the recovered VisA evaluations. Compare experiment gains with the local baseline; paper gaps are unpaired external comparisons. These rows do not establish statistical significance or eliminate trainable-capacity confounds.

[Category comparisons](experiment_categories.md) · [Dataset comparison CSV](experiment_comparison.csv) · [Category comparison CSV](experiment_category_comparison.csv)

Local source: `../runs/baseline/dinov3/0_shot/audit_20261106/family_results.csv`. Rebuild offline with `python scripts/build_experiment_report.py`; no paper parsing, download, training, or evaluation is performed.
