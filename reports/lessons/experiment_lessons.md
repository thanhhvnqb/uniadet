
- [Shared backbone/seed artifact layout](../../runs/maintenance/output_layout/20261105/lessons.md): source grouping, per-target aggregation, collision protection; implementation validation only.

- [Ablation default-target correction](../../runs/baseline/dinov3/0_shot/lessons.md): inspect exact launcher defaults and immutable target lists; validated checkpoint reuse preserves completed training.

- [DINOv3 feature ablation completion audit](../../runs/feature_ablation/reports/20261106_zero_shot_audit/detail_report.md): reciprocal VisA/MVTecAD zero-shot, three seeds; 33 complete / 3 failed / 6 queued. Baseline remains the general reference; preserve missing evaluations and control added head capacity. Per-experiment evidence and lessons are linked from the report.

- [Failed-run recovery and DINOv3 baseline audit](../../runs/baseline/dinov3/0_shot/audit_20261106/lessons.md): output-reader loss can terminate both worker and scheduler; three exact-checkpoint recoveries complete. Upstream features match on tested inputs. Unsmoothed image peaks help VisA but slightly hurt MVTecAD; require source-only selection before a baseline change.

- [Versioned paper reference and comparison ordering](../../runs/maintenance/paper_reference/20261106/lessons.md): cache reported dataset/category metrics with provenance and inconsistencies; use paper → reproduced baseline → experiments, and verify local protocol before joining. Reporting validation only; no benchmark run.

- [Training/testing reproduction review](../../runs/baseline/reproduction_review/20261106_code_review/lessons.md): preserve training outputs; verify pretrained identity; CPU-back exact-NN banks; isolate image-peak smoothing; source-holdout and independent benchmark commands. 210 tests and a real DINOv3 single-seed smoke passed; no full benchmark run.

- [Consolidated results and source-score validation](../../runs/feature_ablation/reports/20261106_consolidated/lessons.md): 36 completed cross-dataset evaluations plus six separate holdout evaluations; raw image peak loses all three holdout seeds, retain baseline; spatial remains unrun.

- [Complete reporting archive and reproduction-gap audit](../../runs/feature_ablation/reports/20261106_consolidated/lessons.md): detailed docs under reports/feature_ablation and reports/reproduction_gap; context archive lacks 84 raw paths, distinguish missing provenance from accuracy causes; reporting only.

- [Feature-ablation cleanup](../../runs/maintenance/feature_cleanup/20261106/lessons.md): active reproduction head and scheduler separated from archived descriptor experiments; baseline checkpoint compatibility retained. Implementation validation only.
