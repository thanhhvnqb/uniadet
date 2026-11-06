# UniADet paper result reference — arXiv 2601.05552v1

Read this cache before reopening the paper. It contains UniADet results for CLIP ViT-L/14@336, DINOv2-R ViT-L/14, and DINOv3 ViT-L/16: **96 dataset/backbone/shot rows and 876 industrial category/backbone/shot rows**, covering all 14 datasets. Industrial shot counts are 0/1/2/4; medical results are zero-shot only. All available seven metrics are retained.

Sources: [paper HTML](https://arxiv.org/html/2601.05552v1), [PDF](https://arxiv.org/pdf/2601.05552v1). Dataset/category values use detailed Tables 12–34; main Tables 2/3/5/10/11 are cached independently for comparison. Every numeric row in detailed Tables 12–34 was independently checked against PDF text: **946 rows matched**, including dataset means and medical domain means.

## Files and conventions

- [Dataset results](dataset_results.csv): seven metric values, published ± values, original numeric cell strings, source table and URL; missing medical tasks remain blank.
- [Category results](category_results.csv): 73 industrial categories × three backbones × four shot counts.
- [Machine-readable reference](reference.json): dataset/category values, main-table values, source identity and discrepancy records.
- [Main-table results](main_table_results.csv), [main/detail discrepancies](discrepancies.csv), [macro discrepancies](macro_discrepancies.csv), and [provenance](provenance.json).
- Readable category tables: [CLIP](categories/clip.md), [DINOv2](categories/dinov2.md), [DINOv3](categories/dinov3.md).

Values are percentages with the precision printed by the paper. Published ± values are not assigned local seed counts or assumed to use ddof=1. A missing paper standard deviation is unavailable, not zero. Dataset Mean rows are stored as macro_mean, never counted as categories. Category labels and numeric values are preserved; MVTec/MVTecAD and RealIAD/Real-IAD dataset names are normalized for repository joins. KSDD has a single category; its category row supplies the dataset result. Medical tables report datasets, not internal category subdivisions.

The method trains on labeled auxiliary VisA test data, using MVTecAD test data as the auxiliary source when evaluating VisA. Keep backbone, source/target protocol and shot regime matched. AUPR must not be silently substituted with AP. Preserve the local evaluator's definitions when comparing.

## Published inconsistencies

Eight main/detail numeric-cell discrepancies occur for DINOv2 Real-IAD 2/4-shot in Tables 10/11 versus Table 32. Five DINOv3 Real-IAD 2-shot Table 34 macro metrics differ from means of its rounded category rows by more than 0.11 pp; the PDF confirms the printed values. Both published values and computed rounding checks remain explicit. Do not silently repair the paper. Full-metric comparisons default to detailed-table dataset rows and identify that source; reproducing a main-table comparison must explicitly use main_table_results.csv instead.

Tables 27/28 repeat the dagger in the second model-column header; the column is identified as DINOv2 by the consistent model order and Table 3 values. Table 2's DINOv2 patch-size caption conflicts with the implementation details; the intended DINOv2-R ViT-L/14 backbone identity is recorded.

## Paper dataset results

These are external paper-reported values, not local reproductions. Each backbone/shot regime is shown separately.

### clip, shots=0

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 92.4 | 96.1 | 92.9 | 91.8 | 42.8 | 44.1 | 88.2 | 12 |
| VisA | 88.0 | 89.1 | 84.5 | 95.8 | 28.0 | 33.3 | 91.0 | 15 |
| BTAD | 96.4 | 97.8 | 93.8 | 95.7 | 42.8 | 48.2 | 81.3 | 18 |
| DTD | 98.1 | 99.1 | 97.1 | 98.4 | 64.3 | 61.2 | 95.2 | 21 |
| KSDD | 96.3 | 83.4 | 80.0 | 98.1 | 31.8 | 43.6 | 86.3 | 24 |
| HeadCT | 97.0 | 96.7 | 91.8 | — | — | — | — | 27 |
| BrainMRI | 95.7 | 96.4 | 95.1 | — | — | — | — | 27 |
| Br35H | 97.4 | 97.1 | 93.0 | — | — | — | — | 27 |
| ISIC | — | — | — | 92.6 | 84.2 | 76.8 | 87.8 | 28 |
| ColonDB | — | — | — | 84.0 | 36.2 | 40.8 | 79.2 | 28 |
| ClinicDB | — | — | — | 90.1 | 55.8 | 55.6 | 81.1 | 28 |
| Endo | — | — | — | 90.7 | 62.5 | 61.5 | 79.3 | 28 |
| Kvasir | — | — | — | 87.5 | 57.5 | 58.6 | 59.7 | 28 |
| Real-IAD | 78.6 | 76.7 | 70.9 | 96.6 | 33.6 | 38.5 | 90.5 | 29 |

### clip, shots=1

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 96.3±0.7 | 98.1±0.3 | 95.6±0.2 | 95.6±0.3 | 55.7±1.3 | 56.4±1.3 | 91.7±0.4 | 12 |
| VisA | 89.3±1.9 | 90.6±1.5 | 86.1±1.6 | 97.5±0.0 | 32.8±0.4 | 41.0±0.8 | 90.8±0.5 | 15 |
| BTAD | 96.6±0.3 | 97.4±0.4 | 93.7±0.5 | 96.2±0.1 | 54.4±2.5 | 55.9±1.6 | 82.6±0.4 | 18 |
| DTD | 98.0±0.2 | 99.1±0.1 | 97.2±0.4 | 97.4±0.1 | 73.4±0.9 | 70.3±0.7 | 94.0±0.2 | 21 |
| KSDD | 97.7±0.7 | 92.4±3.2 | 89.7±2.7 | 98.0±0.0 | 45.0±5.8 | 50.9±3.9 | 87.0±0.1 | 24 |
| Real-IAD | 82.5±0.6 | 80.9±0.6 | 74.0±0.5 | 98.0±0.0 | 35.3±0.2 | 40.6±0.1 | 91.7±0.3 | 29 |

### clip, shots=2

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 97.0±0.6 | 98.3±0.2 | 96.3±0.1 | 95.9±0.1 | 57.0±0.9 | 57.5±0.8 | 92.3±0.3 | 12 |
| VisA | 92.3±0.5 | 93.0±0.4 | 88.1±0.5 | 97.8±0.0 | 35.8±1.0 | 44.0±1.0 | 91.6±0.2 | 15 |
| BTAD | 96.5±0.2 | 97.3±0.4 | 93.8±0.8 | 96.4±0.1 | 54.5±2.1 | 55.9±1.3 | 82.5±0.8 | 18 |
| DTD | 98.0±0.1 | 99.1±0.1 | 97.3±0.3 | 97.6±0.0 | 74.3±0.7 | 70.7±0.5 | 94.3±0.1 | 21 |
| KSDD | 97.7±0.6 | 93.3±1.3 | 89.4±3.6 | 98.1±0.2 | 46.1±3.9 | 50.6±3.6 | 87.4±0.6 | 24 |
| Real-IAD | 83.6±0.2 | 82.0±0.3 | 74.9±0.2 | 98.2±0.0 | 36.4±0.3 | 41.7±0.4 | 92.5±0.2 | 30 |

### clip, shots=4

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 97.7±0.3 | 98.6±0.2 | 96.6±0.2 | 96.3±0.1 | 58.8±0.5 | 59.3±0.4 | 92.9±0.1 | 12 |
| VisA | 93.3±0.2 | 93.8±0.1 | 88.9±0.3 | 98.0±0.1 | 36.7±1.3 | 44.9±0.9 | 92.0±0.1 | 15 |
| BTAD | 96.7±0.2 | 97.8±0.4 | 94.8±1.0 | 96.6±0.1 | 56.5±1.4 | 57.1±1.0 | 82.3±0.2 | 18 |
| DTD | 98.4±0.4 | 99.2±0.2 | 97.7±0.4 | 97.7±0.1 | 75.2±0.3 | 71.3±0.3 | 94.5±0.1 | 21 |
| KSDD | 97.3±0.8 | 91.3±1.8 | 87.1±3.6 | 98.1±0.2 | 42.5±5.7 | 47.9±5.3 | 87.4±0.7 | 24 |
| Real-IAD | 84.3±0.2 | 82.6±0.2 | 75.4±0.1 | 98.3±0.1 | 37.2±0.2 | 42.5±0.3 | 93.0±0.2 | 30 |

### dinov2, shots=0

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 93.5 | 97.2 | 93.8 | 92.4 | 50.9 | 49.7 | 90.1 | 13 |
| VisA | 91.3 | 91.0 | 88.2 | 95.9 | 32.7 | 36.4 | 92.4 | 16 |
| BTAD | 94.4 | 96.3 | 93.0 | 97.7 | 60.6 | 59.2 | 86.1 | 19 |
| DTD | 94.3 | 98.2 | 94.7 | 98.3 | 71.9 | 67.4 | 94.0 | 22 |
| KSDD | 91.6 | 59.5 | 73.0 | 98.4 | 15.8 | 24.5 | 90.2 | 25 |
| HeadCT | 95.2 | 94.9 | 89.2 | — | — | — | — | 27 |
| BrainMRI | 91.7 | 94.8 | 88.0 | — | — | — | — | 27 |
| Br35H | 92.5 | 93.9 | 85.7 | — | — | — | — | 27 |
| ISIC | — | — | — | 95.0 | 88.5 | 81.4 | 90.4 | 28 |
| ColonDB | — | — | — | 86.0 | 35.5 | 43.1 | 79.8 | 28 |
| ClinicDB | — | — | — | 91.3 | 59.3 | 60.6 | 82.2 | 28 |
| Endo | — | — | — | 92.6 | 67.3 | 66.4 | 82.7 | 28 |
| Kvasir | — | — | — | 91.1 | 65.2 | 65.3 | 53.1 | 28 |
| Real-IAD | 82.5 | 80.3 | 73.7 | 97.7 | 43.1 | 45.0 | 93.5 | 31 |

### dinov2, shots=1

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 97.6±0.2 | 98.6±0.3 | 96.5±0.5 | 97.1±0.1 | 63.1±0.5 | 61.7±0.4 | 94.2±0.2 | 13 |
| VisA | 95.2±0.8 | 95.6±0.6 | 91.4±0.5 | 98.2±0.1 | 42.1±0.2 | 47.6±0.0 | 93.3±0.5 | 16 |
| BTAD | 96.4±0.3 | 97.7±0.4 | 93.8±1.3 | 98.4±0.0 | 72.7±0.7 | 68.6±0.6 | 83.7±0.5 | 19 |
| DTD | 98.8±0.1 | 99.5±0.0 | 98.2±0.3 | 98.9±0.1 | 82.4±0.3 | 76.9±0.5 | 96.8±0.2 | 22 |
| KSDD | 94.7±0.4 | 68.7±2.0 | 78.7±1.5 | 99.4±0.3 | 34.9±2.7 | 46.1±0.9 | 97.7±0.6 | 25 |
| Real-IAD | 88.7±0.5 | 86.8±0.3 | 79.3±0.4 | 98.9±0.0 | 48.4±0.4 | 51.1±0.3 | 94.6±0.4 | 31 |

### dinov2, shots=2

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 98.0±0.3 | 98.9±0.1 | 97.2±0.2 | 97.4±0.1 | 64.1±0.6 | 62.6±0.5 | 94.7±0.1 | 13 |
| VisA | 96.1±0.8 | 96.3±0.8 | 92.5±0.4 | 98.4±0.0 | 44.2±0.3 | 49.6±0.2 | 94.1±0.3 | 16 |
| BTAD | 96.8±0.4 | 98.2±0.4 | 95.1±0.7 | 98.4±0.0 | 73.1±0.5 | 68.8±0.4 | 83.8±0.4 | 19 |
| DTD | 99.2±0.1 | 99.6±0.0 | 98.7±0.2 | 99.0±0.1 | 83.0±0.1 | 77.2±0.2 | 97.1±0.2 | 22 |
| KSDD | 95.3±0.2 | 71.2±1.2 | 78.8±1.4 | 99.5±0.1 | 36.9±2.7 | 47.7±1.6 | 97.6±0.3 | 25 |
| Real-IAD | 89.8±0.4 | 87.8±0.3 | 80.4±0.6 | 99.0±0.1 | 49.0±0.6 | 51.1±0.1 | 95.4±0.2 | 32 |

### dinov2, shots=4

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 98.7±0.1 | 99.2±0.1 | 98.0±0.2 | 97.7±0.1 | 65.4±0.2 | 63.6±0.2 | 95.2±0.1 | 13 |
| VisA | 96.9±0.3 | 96.8±0.3 | 93.5±0.1 | 98.6±0.1 | 45.2±0.6 | 50.5±0.3 | 94.5±0.2 | 16 |
| BTAD | 96.9±0.3 | 98.4±0.3 | 95.3±1.1 | 98.4±0.1 | 73.9±0.5 | 69.2±0.4 | 83.7±0.3 | 19 |
| DTD | 99.7±0.1 | 99.7±0.0 | 99.3±0.1 | 99.2±0.1 | 83.8±0.1 | 77.6±0.1 | 97.5±0.2 | 22 |
| KSDD | 95.3±0.6 | 71.7±1.9 | 79.2±1.1 | 99.4±0.2 | 35.2±5.3 | 45.6±5.1 | 96.8±1.5 | 25 |
| Real-IAD | 90.6±0.1 | 88.5±0.1 | 81.4±0.2 | 99.2±0.1 | 49.2±0.1 | 51.0±0.0 | 95.7±0.1 | 32 |

### dinov3, shots=0

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
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

### dinov3, shots=1

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 97.2±0.1 | 98.1±0.3 | 96.5±0.1 | 97.4±0.1 | 66.5±0.9 | 63.6±0.7 | 94.5±0.2 | 14 |
| VisA | 95.8±0.8 | 95.5±0.5 | 92.5±0.8 | 98.2±0.1 | 44.0±0.1 | 49.3±0.3 | 93.6±0.6 | 17 |
| BTAD | 97.1±0.5 | 98.5±0.2 | 96.1±0.3 | 98.3±0.1 | 71.7±1.0 | 66.7±0.8 | 86.1±0.4 | 20 |
| DTD | 99.4±0.1 | 99.6±0.0 | 99.3±0.1 | 99.1±0.0 | 85.8±0.2 | 78.9±0.4 | 97.3±0.1 | 23 |
| KSDD | 96.9±0.3 | 84.7±2.0 | 83.4±0.9 | 99.3±0.1 | 45.2±3.7 | 50.4±2.4 | 94.7±0.3 | 26 |
| Real-IAD | 86.9±0.6 | 85.3±0.5 | 77.9±0.6 | 98.6±0.1 | 47.6±0.5 | 49.4±0.4 | 94.0±0.3 | 33 |

### dinov3, shots=2

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 97.6±0.3 | 98.6±0.2 | 96.8±0.1 | 97.7±0.1 | 67.5±1.0 | 64.4±0.8 | 95.0±0.0 | 14 |
| VisA | 96.8±0.4 | 96.3±0.4 | 93.7±0.4 | 98.3±0.0 | 45.3±0.4 | 50.8±0.3 | 94.0±0.4 | 17 |
| BTAD | 97.3±0.6 | 98.6±0.2 | 96.5±0.2 | 98.4±0.0 | 71.8±1.3 | 66.7±0.9 | 86.2±0.7 | 20 |
| DTD | 99.6±0.1 | 99.7±0.1 | 99.5±0.1 | 99.2±0.1 | 86.1±0.2 | 79.2±0.3 | 97.5±0.0 | 23 |
| KSDD | 97.2±0.4 | 87.5±0.9 | 84.0±1.3 | 99.3±0.2 | 45.0±2.2 | 49.9±1.2 | 94.6±0.4 | 26 |
| Real-IAD | 87.2±1.2 | 85.0±2.2 | 77.4±1.9 | 98.9±0.2 | 47.0±2.9 | 49.0±2.3 | 94.8±0.2 | 34 |

### dinov3, shots=4

| Dataset | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO | Source table |
|---|---|---|---|---|---|---|---|---|
| MVTecAD | 98.2±0.1 | 98.8±0.1 | 97.5±0.1 | 97.9±0.0 | 69.0±0.4 | 65.5±0.4 | 95.6±0.2 | 14 |
| VisA | 97.1±0.1 | 96.6±0.1 | 94.1±0.3 | 98.4±0.1 | 45.5±0.6 | 51.3±0.4 | 94.1±0.2 | 17 |
| BTAD | 97.5±0.3 | 98.8±0.2 | 96.9±0.6 | 98.4±0.0 | 72.7±1.0 | 67.4±0.9 | 86.2±0.4 | 20 |
| DTD | 99.7±0.1 | 99.7±0.1 | 99.6±0.1 | 99.3±0.1 | 86.7±0.1 | 79.6±0.1 | 97.7±0.2 | 23 |
| KSDD | 96.7±0.2 | 86.5±1.7 | 82.4±1.4 | 99.3±0.2 | 44.0±5.7 | 49.3±4.1 | 93.9±2.3 | 26 |
| Real-IAD | 88.5±0.2 | 87.0±0.2 | 79.2±0.3 | 98.9±0.1 | 49.8±0.2 | 51.2±0.2 | 95.2±0.1 | 34 |
