# Training/testing reproduction review

[Full review and experiment commands](../runs/baseline/reproduction_review/20261106_code_review/discussion.md).

## 1. Paper-reported results

| Target | I-AUROC | P-AUPR |
|---|---|---|
| MVTecAD | 94.0 | 52.7 |
| VisA | 91.9 | 32.5 |
| Real-IAD | 81.2 | 41.6 |

Source: [cached DINOv3 zero-shot reference](paper/uniadet_2601.05552v1/README.md).

## 2. Reproduced baseline

| Target | I-AUROC % ± SD | P-AUPR % ± SD |
|---|---|---|
| MVTecAD | 92.47 ± 0.27 | 51.61 ± 0.14 |
| VisA | 88.91 ± 0.53 | 33.42 ± 0.42 |

No completed three-seed Real-IAD baseline is claimed. Exact paper-score reproduction remains unverified.

## 3. Code review and new experiments

Repaired training-output overwrite protection, CPU-backed exact-NN memory, and frozen-weight provenance checks. Added a backward-compatible independent image-score smoothing control and a three-seed source-validation/benchmark launcher. Defaults preserve the historical baseline.

210 tests passed; a real DINOv3 train → save → load → evaluate smoke passed with identical pixel metrics across score variants. No new full benchmark was run. Start with source validation, then freeze a rule before an independent Real-IAD benchmark. [Commands and limitations](../runs/baseline/reproduction_review/20261106_code_review/discussion.md#commands-to-run-next).
