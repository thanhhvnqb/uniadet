# Context ablations

The existing baseline remains the default. These are experimental hypotheses, not validated improvements. The implementation follows the supplied proposal alongside the existing UniADet reproduction (https://arxiv.org/abs/2601.05552).

Run from the repository root using the environment that contains the project dependencies. Each command runs jobs sequentially and skips completed checkpoints/evaluations. Defaults: CLIP, layers 12/15/18/21/24, 15 epochs, Adam lr 1e-3, seeds 10/20/30, VisA→MVTec and MVTec→VisA. Training uses the auxiliary dataset's test split as in the existing reproduction protocol.

```bash
# 0. Reproduce and measure the baseline first
.venv/bin/python scripts/run_context.py baseline
# 2. Inference-only within-image self-similarity
.venv/bin/python scripts/run_context.py self
# 1. Residual b/c/d/e; includes a matched control without mosaic mixing
.venv/bin/python scripts/run_context.py residual
# 3. Appearance vs contextual 1-shot memory, then spatial-window matching
.venv/bin/python scripts/run_context.py memory
# 4. Original vs normal/random/query mosaics, with four-corner TTA
.venv/bin/python scripts/run_context.py mosaic
# Or run all in that order, sequentially
.venv/bin/python scripts/run_context.py all
```

Append `--visa /path/to/VisA --mvtec /path/to/MVTecAD` for other roots. `--dry-run` prints commands without running them. `--seeds 10` provides a quick pilot; use the default three seeds for trained comparisons. `--device cpu`, `--workers 0`, and `--batch-size 1` are available. Commands automatically train a missing baseline before inference experiments. Existing results are reused only within the selected output directory: use a new `--output` if changing data or implementation.

Results include per-category and aggregate metrics in `metrics.json`, prediction scores, effective configurations, and reference selections. Compare pixel AUPR first and image AUROC second. Baseline targets supplied in the proposal are MVTec 92.4/42.8 and VisA 88.0/28.0; no target reproduction or gain is claimed until the actual jobs finish. Training reports the actual number of trainable parameters.

## Individual options

- Residuals: `train.py --context-scales near mid global`, optionally `--context-layers 12 15`, `--context-shared`, or `--residual-cosine`. Default residuals retain magnitude through division by the original feature norm. Each layer/range has its own two-class weights, learned scale, and bias; shared mode shares these across ranges within each layer. Classification stays unchanged. Focal and Dice supervise individual branches, each fused level, and the final fused map. Evaluation averages branch probabilities per layer. Architecture settings restore from checkpoints and reject mismatched evaluation overrides.
- Residual training disables CAA mosaic mixing and retains crop CAA, avoiding context across different images for every range. The runner adds a no-mosaic baseline control. This differs from the original baseline augmentation and must be reported. Near and mid radii are fixed at (0,2] and (2,6] in patch coordinates; global is the whole feature grid.
- Self-similarity: `test.py --self-weight 0.2 --self-radius 3 --self-k 5`; optional `--self-norm-sigma 3` excludes norm outliers as queries and candidates. Matching is within each image, excludes the Chebyshev near neighborhood, and chunks queries to bound working memory. If fewer than k candidates exist, average available neighbors; no candidates gives a neutral zero distance. Layer maps are standardized, averaged, min-max scaled per image, and fused before normal-memory fusion/smoothing. Modified maps use their fused maximum for image scoring.
- Contextual memory: `test.py --shots 1 --memory-beta 0.5`, optionally `--position-radius 8`. Both query and reference use normalized concatenated appearance/near/mid blocks and one joint cosine NN. The radius is in patch coordinates; omission means unrestricted. Existing lambda_f/lambda_p defaults remain 0.5. Reference banks retain category/view separation.
- Mosaic: `test.py --inference-mosaic normal --mosaic-tta` (or `random`, `query`). Uses a fixed canvas at the checkpoint image size, so tiles lose resolution. Normal uses training-split normal references, random uses seeded unverified test images from the same category/view, query duplicates the query. Random is an explicitly transductive control and may include the query itself. Selections are saved. Normal reference sampling uses `max(1, shots)`; `shots=0` still uses one reference for mosaic. Memory scoring is bypassed during mosaic inference: the normal references serve only as canvas tiles. Query maps are cropped and resized; image scoring combines the canvas classifier with the query-map peak. Four corners are averaged. Mosaic still smooths on the canvas before cropping. Run separately from contextual memory.

The runner uses illustrative fixed hyperparameters; tune these only on held-out auxiliary categories, never the target test sets. Category filters (`--categories`, `--exclude-categories`) are available on train.py/test.py for such splits. No automatic target-based model selection is implemented. The runner covers the two cross-dataset protocols; evaluate additional prepared datasets with test.py using the VisA-trained checkpoint (except VisA itself). Track texture categories, KSDD, and medical datasets separately before expanding experiments.
