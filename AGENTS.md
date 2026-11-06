# Agent instructions for UniADet

## Language

- Always respond in English, regardless of whether the user writes in Vietnamese or English.
- Write all new code, comments, documentation, logs, and other generated output in English.

## Scope and reasoning

- This file contains durable coding, reasoning, and execution instructions only.
  Store experimental findings, session lessons, failed hypotheses, and historical
  comparisons in the corresponding experiment directory under `runs/<direction>/`,
  never in this file. Keep the reusable lesson index under `reports/lessons/`.
  Versioned paper reference data and cross-experiment comparison reports live
  under `reports/`, as specified in the reporting rules below.
- Before changing an experiment, read the relevant reports and
  `reports/lessons/experiment_lessons.md`. Check that their dataset, backbone,
  shot count, and other conditions apply before reusing a conclusion.
- Treat proposed improvements as hypotheses until supported by completed,
  controlled experiments. Separate observations, explanations, and speculation.
  A higher mean alone does not establish statistical significance.
- Read `git status` before editing and preserve existing user changes. Do not
  overwrite or delete checkpoints, logs, or previous results to rerun a job.
- Implement and evaluate each component independently before combining components.
  Run the experiment family requested by the user; do not launch every available
  research direction automatically.

## Coding constraints

- Keep the backbone frozen: `requires_grad_(False)`, `eval()` mode, and feature
  extraction without gradients. Do not add backbone adapters or fine-tune the
  encoder. A trainable feature module inside the head is a separate architecture
  change; report its parameter count and use an appropriate capacity control.
- Preserve baseline behavior when new features are disabled. In particular,
  the baseline head must preserve the original formula, parameter count,
  and state dictionary compatibility.
- Keep configuration explicit. Do not introduce hidden behavior based on MODEL
  names or experiment names, or mutate global configuration while scheduling runs.
- New settings need validation, serialization, and checkpoint restoration rules.
  Evaluation settings set to `None` inherit checkpoint values. Missing fields in
  older checkpoints use historical defaults, not current training defaults.
  Record every explicit inference override in the evaluation configuration.
- Retrain the head when its feature representation changes. Inference-only changes
  may reuse a compatible checkpoint, with separate outputs and explicit overrides.
- Compute zero-shot descriptors independently for each query image. Fit feature
  normalization statistics on source training data, save them in checkpoints,
  and keep them fixed during evaluation. Do not use target labels or masks as
  descriptor inputs or fit statistics across target images.
- Handle image boundaries, empty neighborhoods, empty clusters, zero norms, and
  memory limits explicitly. Use chunked exact kNN when a full `B × N × N` tensor
  is unnecessary. Cluster identifiers are not numerical classifier features.
- Keep descriptor groups independent during initial ablations. For example,
  `components` may use clustering labels to compute connected-component areas,
  but must not silently concatenate the separate `cluster` descriptor group.
- Preserve valid anomaly supervision after crop, resize, and mosaic augmentation.
  Keep `preserve_anomaly_mask=True` for new runs; reject or fall back from an
  augmentation that erases the anomaly mask of a positive training example.
- For background filtering, match UniADet's processing resolution and measure both
  background removal and accidental foreground/anomaly removal. Target anomaly
  masks may be used for evaluation, never to guide the inference algorithm.

## Default evaluation protocol

- Train on **VisA** and validate on **Real-IAD**. Keep these defaults in `config.py`:
  `TRAIN_CONFIG["data"] = "datasets/VisA"` and
  `TEST_CONFIG["data"] = ["datasets/Real-IAD"]`.
- Use VisA's labeled auxiliary `test` split for training, following the existing
  reproduction protocol. Do not replace it with the normal-only `train` split.
  The current evaluator reads the target dataset's `test` split.
- Run every baseline and compared configuration with **three seeds: 10, 20, 30**,
  unless the user specifies another set. Match evaluation and training seeds;
  record reference-selection seeds separately when applicable.
- Single-seed or shortened runs are smoke tests or pilots, not completed benchmarks.
  Label incomplete runs and never present missing seeds as completed observations.
- Use 15 training epochs and `epoch_15.pt` as the default comparison checkpoint.
  Fix the checkpoint before target evaluation; do not select the best target epoch.
- Select new hyperparameters on held-out VisA categories excluded from both training
  and CAA sampling. Do not use Real-IAD labels to fit normalization, thresholds,
  or select the final configuration. If Real-IAD is deliberately used for tuning,
  disclose it as development data and evaluate generalization on independent data.
- Keep the backbone, pretrained weights, layers, resolution, optimizer, batch size,
  augmentation, loss, and postprocessing matched except for the factor under study.
  Use separate baselines for different backbones and evaluation protocols.
- Control confounds explicitly: changes to mosaic augmentation, layer coverage,
  loss-branch count, trainable capacity, or TTA require matched controls before
  attributing a gain to the proposed feature.

## Keep zero-shot and one-shot separate

- **Zero-shot (`shots=0`)** uses no normal target reference images. Source-dataset
  supervision is allowed under the protocol above; target supervision is not.
- **One-shot (`shots=1`)** uses one normal target training reference per category/view,
  following the repository's reference-selection convention. Record selected
  references, view grouping, and sampling seeds.
- Compare zero-shot variants only with a zero-shot baseline, and one-shot variants
  only with a one-shot baseline using the same references and sampling seeds.
  Do not report a one-shot result minus a zero-shot baseline as an improvement
  from the proposed method. Keep rankings, aggregates, and conclusions separate.
- Separate tables or clearly separated sections are required for each regime.
  Baseline identifiers, output directories, manifests, and CSV rows must include
  the regime and shot count; never average results across shot counts.
- Classify the regime by actual data access, not only by the `shots` flag. A method
  using normal target images in a mosaic or another inference branch is not
  zero-shot even if the memory-bank flag is 0. Access to other target test images
  is transductive and must be disclosed and evaluated separately.
- Apply the same matched-shot rule to any additional few-shot settings. Do not
  change reference counts or target information access inside a feature ablation.

## Experiment script organization

- Scripts written or modified to run experiments must follow the organization
  of `train_test.py`. Read that file before implementing an experiment runner.
- Keep Python entry points thin: build an invocation-local `workflow.RunConfig`,
  resolve training arguments through `workflow.training_args`, and reuse
  `cli.seed_all`, `cli.train`, and `workflow.test_checkpoints` for execution.
  Keep shared orchestration in `uniadet/workflow.py`; do not duplicate training,
  checkpoint discovery, evaluation, or metric logic in experiment scripts.
- Organize runners around configuration construction and `main(argv=None)`,
  with argument parsing and an `if __name__ == "__main__"` guard. Support the
  applicable `train`, `test`, and `train_test` modes and a `--dry-run` that prints
  resolved settings and output paths without training or evaluation. Imports
  must not start experiments.
- Apply explicit CLI overrides to the local configuration before resolving
  arguments; apply shared execution settings consistently to training and test.
  Preserve `train_test.py` semantics: seed before execution, train before test
  in `train_test` mode, and use `--start-epoch` only to filter test checkpoints.
- Bash launchers schedule isolated Python runner invocations and collect their
  statuses; Python coordinators may handle plans, snapshots, and aggregation.
  Both must retain this shared execution flow and the artifact layout below.

## Schedule work on three GPUs

- GPU 1 is faulty and must not be probed or scheduled. Use only physical GPUs
  0, 2, and 3; the current protocol uses seeds 10, 20, and 30.

- When multiple jobs are needed, write a **Bash launcher in `scripts/`**. Run up to
  three jobs in parallel, one job per GPU. A job trains and then validates one
  `(configuration, seed)` pair; independent seeds do not need DDP.
- For one configuration, assign seeds 10/20/30 to GPUs 0/2/3 respectively.
  Queue additional configurations with at most one launcher job per GPU at a time.
- Check `nvidia-smi`, existing workloads, available VRAM, and CUDA in the actual
  Python environment first. Use only healthy, available GPUs; queue work if fewer
  than three are usable and still complete all three seeds. Do not terminate
  processes outside the requested experiment or silently fall back to CPU.
- Set `CUDA_VISIBLE_DEVICES=<physical_gpu>` per process and use `--device cuda:0`
  inside it. The selected physical device becomes logical device 0.
- Use `set -euo pipefail`, separate logs, and recorded PIDs, GPUs, seeds, and commands.
  Collect the exit status of every child with `wait`; a bare `wait` is insufficient
  to detect all failures. Preserve pipeline failures when using `tee`. Record all
  job outcomes and return failure if any job fails.
- Create immutable per-job configuration snapshots and unique output paths before
  launching. Do not repeatedly edit shared `config.py` while jobs are reading it.
  Use `scripts/run_baseline_review.py --config <snapshot.py>` or explicit CLI
  settings, verifying their effective configurations match the intended protocol.
- Validate only after successful training and checkpoint verification. Workers must
  not write to a shared `test_summary.csv`; aggregate through one coordinator.
- Provide a launcher dry-run showing GPU, seed, command, input, checkpoint, and
  output paths. Check Bash syntax with `bash -n scripts/<launcher>.sh`.
- Budget DataLoader workers, CPU threads, and RAM across all jobs. Cache pretrained
  weights before launching and use the same weights version for matched runs.

## Artifacts and reporting

- Use the cached paper reference in `reports/paper/uniadet_2601.05552v1/`
  (`README.md`, `dataset_results.csv`, `category_results.csv`, and `reference.json`)
  before reopening or reparsing the paper. Check the paper version, source table,
  backbone, dataset, shot count, metric, and training protocol before comparing.
  Refresh the cache only for a new paper version, missing coverage, or a verified
  extraction error; record the source and preserve reported inconsistencies.
- Present quantitative summaries in this order: **paper-reported results first,
  the locally reproduced matched baseline second, and other experiment results
  third**. Apply this order to dataset and category tables and comparison CSVs;
  put the paper row above the baseline and variants within each matched group.
  Use `reports/experiment_comparison.md` as the current cross-experiment entry
  point and retain experiment-specific evidence under `runs/<direction>/`.
  Rebuild the current comparison offline with
  `python scripts/build_experiment_report.py`; verify the selected input results
  and backbone/shot provenance when reporting a different experiment family.
- Clearly label paper references as external reported values, including their
  published precision and any reported standard deviation. Label local baseline
  values with completed seeds, sample SD, checkpoint, and source artifacts.
  Never describe a diagnostic, tuned target result, missing run, or paper value
  as a reproduced baseline. Missing references remain unavailable rather than
  being substituted from another backbone, shot regime, dataset, or metric.
  Compute paired experiment deltas against the matched local baseline; paper
  gaps are unpaired external comparisons. Keep paper summary/detail discrepancies
  visible rather than silently replacing either published number.
- Store all new experiment outputs under **`runs/<direction>/`**, grouped by
  experiment type and then by a unique experiment ID. Each experiment has its own
  directory; never mix outputs from different experiments or reruns. Use
  `runs/baseline/` for new baseline experiments and the same layout as other
  directions. Preserve historical artifacts in their existing locations.
- Use this required layout (repeat seed directories for 10, 20, and 30):

  ```text
  runs/<direction>/<backbone>/
    0_shot/                 # or 1_shot, 2-shot, etc.
      manifest.json
      configs/
      seed_10/
        train_<dataset_name>/            # checkpoints, training logs, outputs
        test_<dataset_name>/             # raw metrics, test logs, predictions

      results.csv
      summary.csv
      discussion.md
      lessons.md
  ```

- Keep every seed's training outputs in its `train/` directory and its evaluation
  outputs in separate `test/<dataset_name>/` directories for each tested dataset.
  Store immutable per-job configuration snapshots in `configs/` and link them
  from the manifest. Use distinct dataset identifiers when splits or protocols
  differ, so evaluation outputs cannot overwrite each other.
- After all jobs finish, save aggregated results and analysis, when generated,
  in the corresponding experiment's regime directory shown above, within
  `runs/<direction>/`. Keep datasets separate in aggregation and never combine
  shot regimes. Workers write only their own seed outputs; one coordinator writes
  experiment-level `results.csv` and `summary.csv`.
- Manifests must link to real artifacts. Keep small reports and configurations in
  version control, explicitly including them if `runs/` is ignored; do not commit
  datasets, pretrained weights, large checkpoints, or logs.
- Each manifest records the direction/variant, matched baseline, regime, shots,
  seeds, dataset splits/categories, checkpoint epoch, effective configurations,
  commands, weight source, environment/GPU versions, artifact paths, and job status.
  Include the commit, working-tree state, and a snapshot/diff of relevant uncommitted
  changes so the recorded commit is sufficient to identify the actual code used.
- Complete three seeds for each matched baseline. Preserve initial results; create
  a new baseline version when a protocol change or prediction-affecting bug fix
  requires one. Reuse historical artifacts only after checking compatibility and
  provenance. Do not move or rewrite old results just to match a new layout.
- After an authorized experiment finishes, verify completion, aggregate its results,
  and write the analysis. Commands or logs alone do not complete an experiment task.
  If only implementation was requested, clearly state that no benchmark was run.
- `results.csv` contains variant × regime × seed × category rows and a macro row for
  each seed. Include dataset, shots, checkpoint epoch, status, and source artifacts.
  `summary.csv` reports means and **sample standard deviations (`ddof=1`)** over
  the three seeds, paired deltas against a matched-regime baseline, and the number
  of seeds improved. Compute category macro averages within each seed first.
- Primary metrics are **P-AUPR** and **I-AUROC**; also report I-AUPR, I-F1max,
  P-AUROC, P-F1max, and P-AUPRO when available. Preserve definitions in
  `uniadet/evaluation_metrics.py`: AUPR is PR-curve area, not AP; AUPRO is normalized
  over FPR `[0, 0.3]`. JSON uses `[0, 1]`, presentation CSVs use percentages, and
  deltas use percentage points. Round only for presentation.
- Treat `average` and `macro_mean` as aliases for the same aggregate, not categories.
  Do not replace missing/undefined metrics with zero. Report valid category counts,
  completed seed counts, and failed jobs. Check CSV values against raw metrics and
  effective configurations; a final training epoch does not prove evaluation ended.
- Explain the idea, implementation, matched controls, results, per-category changes,
  seed stability, time/VRAM/parameter costs, limitations, and next steps in
  `discussion.md`. Measure loading/inference separately from metric computation.
- Persist lessons from each experiment/session under the corresponding
  experiment regime directory under `runs/<direction>/` as `lessons.md`, with
  evidence, applicable protocol, failed
  hypotheses, confounds, and actionable follow-ups. Maintain links to reusable
  lessons in `reports/lessons/experiment_lessons.md`. Keep experimental narratives
  and measured outcomes out of `AGENTS.md`.

## Code map and validation

| Responsibility | Main file |
|---|---|
| Default configuration | `config.py` |
| Training/evaluation and CLI | `uniadet/cli.py` |
| Setting validation and checkpoint restoration | `uniadet/settings.py` |
| Orchestration, paths, checkpoint discovery | `uniadet/workflow.py` |
| Frozen feature extraction | `uniadet/backbones.py` |
| Decoupled heads, memory, prediction | `uniadet/model.py` |
| Data loading and augmentation | `uniadet/data.py` |
| Losses and metric definitions | `uniadet/loss.py`, `uniadet/evaluation_metrics.py` |

- Run relevant tests for model/data/settings changes and the full suite before a
  long benchmark. Validate resolved settings through dry-run, then smoke-test
  train → save → load → evaluate on a small subset before a three-GPU launch.
- Verify that referenced scripts exist and inspect `--help` before reusing commands
  from reports. The reproduction launcher prints a plan without `--execute`. `--start-epoch` filters evaluation checkpoints;
  it does not resume training or restore optimizer progress.
- Use a new output directory for a new training run to avoid checkpoint overwrite
  and mixed append-only logs. Do not change the evaluation protocol merely to get
  a failing test to pass.

```bash
uv run python -m pytest -q
uv run python train_test.py --dry-run
bash scripts/run_reproduction_review.sh --help
```
