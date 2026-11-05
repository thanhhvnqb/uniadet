# UniADet context experiments: results and discussion

Generated 2026-11-05 from `runs/context`. All 84 evaluations are complete: 14 configurations × 2 cross-dataset directions × 3 seeds (10, 20, 30). All 36 distinct training logs contain epochs 1–15. Each evaluation reports checkpoint epoch 15.

The strongest result is contextual 1-shot memory: it improves both primary P-AUPR and I-AUROC over appearance-only 1-shot matching on both datasets. Trained residual branches fail to improve over the matched augmentation control. Self-similarity helps VisA but hurts MVTec overall. Normal-reference mosaics improve the original baseline, but their advantage over unverified references is dataset-dependent.

## Understand the experiments before reading the scores

Imagine inspecting a circuit board. A small component may be defective because it looks unusual, because it differs from nearby components, or because it looks correct but is surrounded by the wrong things. These experiments test different ways to provide that context to UniADet.

The examples below explain the hypotheses. They are illustrations, not additional measured results.

### Vocabulary and the common starting point

- **Query:** the image currently being inspected.
- **Patch:** one location on the backbone's feature grid. Each location has a vector of numbers describing visual content. A feature vector is more than an RGB color; it can encode shape, texture, and object information.
- **Layer:** an intermediate stage of the CLIP image encoder. We read stages 12, 15, 18, 21, and 24. Each stage produces its own description of the image.
- **Head:** the small trainable component that turns these feature vectors into normal/anomalous scores. The CLIP encoder stays frozen in every experiment here.
- **Anomaly map:** a heatmap showing which locations look defective. A high value means more anomalous.
- **Reference:** a separate example image used for comparison. Normal references are known to be defect-free.
- **Memory bank:** stored feature vectors from normal references. It is a collection of examples, not another trained neural network.
- **Residual:** a difference vector, here `patch feature − average context feature`.
- **Control:** an experiment that keeps other conditions comparable so we can identify the contribution of a particular change.

The baseline learns a normal direction and an anomaly direction for patch features at each selected layer. At test time, it asks how strongly each patch points toward those directions. It also has separate heads that classify the whole image. Patch probabilities are averaged across layers to create the anomaly map.

CLIP's features already mix information through attention. Calling the baseline “patch scoring” means the final segmentation head scores each patch separately; it does not mean the feature encoder has never seen the surrounding image.

```text
Query image → frozen CLIP → patch features → trained patch heads → anomaly map
                         ↘ whole-image features → trained image heads → image score
```

The final image score combines the image-head prediction and a peak from the anomaly maps. Thus localization changes can also change image-level results.

**What does zero-shot mean here?** The heads were trained on a different, auxiliary dataset. When evaluating MVTec, training used VisA; when evaluating VisA, training used MVTec. The zero-shot baseline receives no normal example from the target dataset at inference. The memory experiments do receive target normal examples, so they are few-shot. The normal-reference mosaic also uses a target normal example, even though its saved memory-bank `shots` flag is zero.

### The four ideas differ in what they compare

| Experiment family | The question asked about a patch | Where context comes from | Train a new head? | Target normal image needed? |
|---|---|---|---|---|
| `baseline` | Does its feature resemble the learned normal or anomaly direction? | CLIP features; no added comparison | Yes, baseline head | No |
| Residuals | How does it differ from its surrounding feature average? | Near, middle, or whole-image averages in the query | Yes, new heads and the original heads | No |
| `self` | Can a similar patch be found elsewhere in this same image? | Distant patches in the query | No; reuse baseline | No |
| Memory | Can a matching patch or patch-plus-context be found in a good example? | Stored patches from a normal reference | No; reuse baseline | Yes, one per category/view |
| Mosaic | Does the model judge the query better when it sees other images on the same canvas? | Images placed beside the query before CLIP runs | No; reuse baseline | Yes for `normal`; no for `query`; `random` uses unverified test images |

Residuals change what the head learns. Self-similarity and memory compare already-extracted features. Mosaics change the input image, so CLIP can produce different features in the first place. These four changes were evaluated separately, not stacked together.

### Idea 1: learn whether a patch disagrees with its surroundings

**Intuition.** In a uniform surface, a scratch may differ from the neighboring material. Instead of giving the head only “what is at this patch,” also give it “how this patch differs from the surrounding area.” An ordinary object boundary can also differ from its surroundings; that is why the residual goes into a learned classifier rather than automatically being called a defect.

For each selected layer, compute an average feature from a spatial region and subtract it from the current patch feature:

```text
current patch feature ───────────────────────────→ original patch head
          │
          └─ subtract context average, divide by
             current feature length ────────────→ extra residual head

average the heads' normal/anomaly probabilities → layer anomaly map
average layer anomaly maps                     → final anomaly map
```

Dividing by the original feature length makes the difference relative to the patch's scale. It does not normalize every residual to length one: a small difference remains small. The residual has a direction as well as a size; the head learns which differences matter.

The regions are measured on the **patch grid**, not in image pixels:

| Range | Which features are averaged? | Interpretation |
|---|---|---|
| Near | A 5×5 square around the patch, excluding the center | Immediate surroundings; square-grid distance greater than 0 and at most 2 |
| Mid | The outer part of a 13×13 square, excluding its central 5×5 square | Broader surroundings; distance greater than 2 and at most 6 |
| Global | All patch locations in the image, including the current location | Whole-image feature average |

At image edges, only the available locations contribute to near/mid averages. “Global” here is an average of patch features, not the encoder's special whole-image token.

**What each saved experiment means:**

| ID | What changes | What to compare it with |
|---|---|---|
| `residual_control` | Baseline head, but disable training mosaics | `baseline`: measures the augmentation change alone |
| `b_near` | Add one near residual head at layers 12 and 15; other layers retain their original patch head | `residual_control`: does the smallest context addition help? |
| `c_mid` | Add near and mid residual heads at all five layers | `residual_control`: does the larger context model help? Compared with `b_near`, both range and layer coverage change |
| `d_global` | Add near, mid, and global residual heads at all five layers | `c_mid`: does adding whole-image context help? |
| `e_shared` | Same input ranges as `d_global`, but reuse one residual head across ranges within each layer | `d_global`: do ranges need separate learned classifiers? |

In `d_global`, a layer has four branches: original appearance, near difference, mid difference, and global difference. Their probabilities have equal weight. In `e_shared`, the three residual inputs use the same weights, scale, and bias at a given layer; the original appearance head remains separate. Heads are still separate across layers.

These runs train the original and extra heads together from initialization. They do not attach an extra head to an already-trained baseline and leave the baseline head fixed. Training supervises the individual branch maps, fused layer maps, and final fused map with focal and Dice losses; whole-image heads keep cross-entropy loss. Focal loss emphasizes difficult pixel predictions, while Dice loss encourages overlap with the defect mask.

**Why the extra control?** Training CAA can stitch several images into one mosaic. A context average could then mix unrelated images. Residual runs therefore disable that stitching, while retaining CAA crops. Comparing against `residual_control` avoids attributing the augmentation change to the residual heads. This training mosaic is distinct from the inference mosaic in Idea 4.

**Reading the outcome.** `b_near` gives 40.57 P-AUPR on MVTec versus 42.74 for the matched control. That is a 2.17-point loss, not an improvement. All residual variants fall below the matched control on both datasets. Shared heads do somewhat better than separate heads here. The additional context did not produce a useful aggregate gain under this training setup; the results do not identify a single cause for the loss.

### Idea 2: use repeated patterns inside the query as evidence

**Intuition.** A normal carpet pattern may repeat throughout the image. A damaged patch might have no good match elsewhere. This approach asks “does anything else in this image look like this?” It does not learn a new classifier and does not use another image.

For each patch, at each layer:

1. Exclude the nearby square of radius three patches, including the patch itself. This prevents a patch from trivially matching itself or an overlapping neighbor.
2. Compare its normalized feature with all remaining patches in the same image using cosine similarity.
3. Take the five most similar eligible patches and average their similarities.
4. Convert similarity into an anomaly score: `1 − average similarity`. A poor best match means a larger score.
5. Standardize each layer's map, average across layers, and rescale the resulting map to 0–1 within that image.
6. Fuse it with the original baseline map: `80% baseline + 20% self-similarity`, then apply the usual smoothing.

A hypothetical patch with average top-five similarity 0.95 gets a raw self-score of 0.05; a patch with similarity 0.60 gets 0.40. These are raw intermediate scores, before per-image normalization and fusion.

**Why it can fail.** A unique but healthy component may have no matching copy in the image and look suspicious. Conversely, a large defect may contain many similar defective patches that match one another. High self-score means “unusual within this image,” which is not always the same as “defective.”

**Saved ID and comparison.** `self` reuses exactly the baseline checkpoint. Compare it with `baseline` to assess the added inference rule. The image-score localization peak also switches to the maximum of the fused map, so image-level changes include that pooling change; P-AUPR directly evaluates the changed heatmap.

**Reading the outcome.** MVTec P-AUPR decreases from 43.10 to 42.21, while VisA increases from 28.58 to 29.57. Even the MVTec texture subset decreases overall. Repetition is a plausible source of evidence, but this fixed rule is not consistently useful across these datasets.

### Idea 3: compare with a known-good example, including its context

**Start with appearance memory.** `memory_appearance` extracts all patch features from one normal reference per target category/view. A query patch searches those stored features and uses its best match. A poor best match produces a high anomaly score. Many normal patches are stored even though only one normal image is supplied: “1-shot” counts images, not patches.

**The weakness being tested.** Imagine a correctly shaped component placed in the wrong neighborhood on a board. Appearance-only matching may find that component somewhere in the normal reference and declare a good match. The proposed contextual key asks the matching location to agree on both the component and its surrounding structure.

For `memory_context`, create one longer descriptor for every query and reference patch:

```text
[ normalized appearance,
  sqrt(0.5) × normalized near difference,
  sqrt(0.5) × normalized mid difference ]
             ↓ concatenate, then normalize
        one descriptor for one joint best-match search
```

A candidate reference patch must explain all three parts **at the same location**. The method does not find one reference location for appearance and a different location for context and then combine those independent matches. Near and mid use the same regions defined for Idea 1.

For nonzero blocks, the relative contributions to cosine similarity are appearance:near:mid = 1:0.5:0.5. This corresponds to 50% appearance and 25% each context block after normalization. Unlike the trained residual heads in Idea 1, these context blocks are individually normalized: this memory variant mainly compares their directions rather than retaining residual size.

The memory distance maps are averaged across layers and fused with the baseline map using equal weights: `50% baseline + 50% memory distance`. This weight is distinct from the beta=0.5 inside the descriptor. None of the memory variants trains additional weights.

| ID | What a candidate normal patch must match | Correct comparison |
|---|---|---|
| `memory_appearance` | Appearance only, anywhere in the normal reference | `baseline` shows the combined effect of adding normal references and memory matching |
| `memory_context` | Appearance, near difference, and mid difference jointly; anywhere in the reference | `memory_appearance` isolates the change of descriptor at the same shot count |
| `memory_position` | The same contextual descriptor, but only within ±8 patch rows and columns of the query's coordinates | `memory_context` tests whether spatial alignment helps |

For example, a query patch at row 10, column 12 in `memory_position` may match reference rows 2–18 and columns 4–20, clipped to grid boundaries. The unrestricted version may search the entire reference grid. A spatial window can prevent a component from matching a similar component on the other side of a board, but it can also reject valid matches when objects move, rotate, or differ in layout.

**Reading the outcome.** On MVTec, contextual keys increase P-AUPR from 51.80 to 53.81; on VisA, from 32.59 to 33.70. This is the cleanest positive comparison because the two methods use the same baseline checkpoint, same normal references, and same fusion weights. The spatial window lowers the dataset averages, although it helps transistor and cable. Improvement is observed; the proposed “wrong neighborhood” mechanism remains an explanation to investigate, not a proven cause of every gain.

### Idea 4: put comparison images beside the query before feature extraction

**Intuition.** Instead of comparing feature vectors after CLIP has run, let the encoder see the query beside other images. The hypothesis is that attention across tiles can make the reference useful. Seeing the images together does not guarantee the encoder actually uses them in the intended way.

One inference pass constructs this canvas:

```text
+-------------------+-------------------+
|                   |                   |
|       QUERY       |    REFERENCE      |
|                   |                   |
+-------------------+-------------------+
|                   |                   |
|    REFERENCE      |    REFERENCE      |
|                   |                   |
+-------------------+-------------------+
```

The canvas stays 518×518, so each tile is 259×259. With one reference image, that same image fills all three reference tiles. Run the original encoder and heads on the canvas, keep only the query tile's anomaly map, and resize it back to the query output size. Repeat with the query in each of the four corners and average its predictions. This is four-position **test-time augmentation (TTA)**: four inference passes per query, without retraining.

| ID | What occupies the other three tiles? | What question does this control answer? |
|---|---|---|
| `mosaic_a` | No canvas; ordinary baseline inference | Original result to beat |
| `mosaic_normal` | One known-good reference from the target training split, repeated | Does seeing a normal example help? |
| `mosaic_random` | One seeded image of the same category/view from the target test set, repeated; its label is not checked | Must the accompanying example be known normal? |
| `mosaic_query` | The query itself, repeated | Can replication and canvas processing account for the change without an external image? |

There is no nearest-neighbor memory search in these mosaic predictions. The other image enters as pixels on the canvas. This is why normal-reference memory and normal-reference mosaics are different experiments even though both consume a good example.

**Reading the outcome.** Normal mosaics increase MVTec P-AUPR from 43.10 to 47.75. However, random mosaics already reach 46.84, so only 0.91 points separate known-normal and unverified companions. On VisA, normal mosaics reach 30.83 while random mosaics reach 31.33. Query duplication loses on both datasets. External companions seem more useful than query duplication in these runs, but known-normal status does not consistently improve localization. The random control has access to other test images, so it uses a different information source from the normal-reference setting.

### How to read one results row without mixing comparisons

Take MVTec `memory_context`, P-AUPR **53.81 ± 0.52**:

- **53.81** is the average of the three seed-level results. Each seed first averages the 15 category scores equally.
- **±0.52** is the sample standard deviation across those seeds, not a confidence interval and not variability across pixels.
- Against the zero-shot baseline **43.10**, it gains **10.71 percentage points**, but that includes receiving a normal target reference.
- Against appearance-only 1-shot memory **51.80**, it gains **2.00 percentage points**. This is the relevant comparison for whether adding context to the memory descriptor helps.

**P-AUPR** measures how well the heatmap ranks defective pixels ahead of healthy pixels across decision thresholds; higher is better. **I-AUROC** measures how well whole-image scores rank defective images ahead of normal images; higher is better. A method can improve one while worsening the other. Neither score is simply “the percentage of predictions that were correct.” Small differences in the last decimal can reflect rounding; all differences in the CSV are computed before rounding.

## Reading the results

- Scores are percentages; differences are percentage points (pp). Tables show mean ± sample standard deviation across seeds (`ddof=1`).
- Each seed first averages categories equally: 15 for MVTec and 12 for VisA. These are category macro means, not pooled-pixel metrics.
- P-AUPR is the primary metric. I-AUROC is secondary. The CSV also contains I-AUPR, I-F1max, P-AUROC, P-F1max, and P-AUPRO.
- Comparisons use paired seeds. Paired-difference standard deviations and counts of improving seeds are in the CSV. Three seeds do not establish statistical significance.
- `mosaic_a` repeats the baseline with the same checkpoint/settings and is not an additional independent trial.

Saved settings: CLIP, input 518×518, layers 12/15/18/21/24, 15 epochs, Adam lr=0.001, Gaussian sigma=4, fusion weights lambda_f=lambda_p=0.5. VisA trains the model evaluated on MVTec; MVTec trains the model evaluated on VisA. The auxiliary training split is `test`, following the implemented paper protocol. Few-shot normal references come from the target training split.

## Dataset-level results

| Configuration | MVTec I-AUROC | MVTec P-AUPR | VisA I-AUROC | VisA P-AUPR |
|---|---:|---:|---:|---:|
| baseline | 91.62 ± 0.33 | 43.10 ± 0.55 | 88.08 ± 0.35 | 28.58 ± 0.26 |
| self | 90.40 ± 0.20 | 42.21 ± 0.89 | 88.48 ± 0.37 | 29.57 ± 0.23 |
| residual_control | 92.19 ± 0.19 | 42.74 ± 0.42 | 87.64 ± 0.22 | 28.42 ± 0.12 |
| b_near | 90.62 ± 0.25 | 40.57 ± 0.70 | 87.72 ± 0.17 | 27.54 ± 0.09 |
| c_mid | 89.90 ± 0.28 | 37.87 ± 0.21 | 87.96 ± 0.10 | 25.46 ± 0.24 |
| d_global | 90.39 ± 0.17 | 39.88 ± 0.26 | 87.51 ± 0.47 | 26.27 ± 0.12 |
| e_shared | 90.70 ± 0.36 | 40.58 ± 0.22 | 87.70 ± 0.12 | 26.45 ± 0.16 |
| memory_appearance | 94.17 ± 0.11 | 51.80 ± 0.45 | 91.16 ± 0.23 | 32.59 ± 0.36 |
| memory_context | 95.00 ± 0.14 | 53.81 ± 0.52 | 91.58 ± 0.24 | 33.70 ± 0.37 |
| memory_position | 94.42 ± 0.37 | 52.65 ± 0.43 | 91.56 ± 0.27 | 33.13 ± 0.30 |
| mosaic_a | 91.62 ± 0.33 | 43.10 ± 0.55 | 88.08 ± 0.35 | 28.58 ± 0.26 |
| mosaic_normal | 93.28 ± 0.54 | 47.75 ± 0.41 | 90.39 ± 0.95 | 30.83 ± 1.06 |
| mosaic_random | 91.35 ± 0.48 | 46.84 ± 0.84 | 85.87 ± 2.05 | 31.33 ± 1.12 |
| mosaic_query | 88.61 ± 0.26 | 37.89 ± 0.42 | 83.91 ± 0.11 | 25.16 ± 0.30 |

## Baseline reproduction

- MVTEC: 91.62 ± 0.33 / 43.10 ± 0.55 I-AUROC / P-AUPR, versus the proposal's paper targets 92.4 / 42.8. Differences: -0.78 / +0.30 pp.
- VISA: 88.08 ± 0.35 / 28.58 ± 0.26 I-AUROC / P-AUPR, versus the proposal's paper targets 88.0 / 28.0. Differences: +0.08 / +0.58 pp.

These runs are close to the supplied targets, but do not exactly reproduce them. The paper targets are taken from the experiment proposal; this report does not independently re-audit the paper table or data preparation.

## Controlled comparisons

| Variant − reference | MVTec ΔI-AUROC | MVTec ΔP-AUPR | VisA ΔI-AUROC | VisA ΔP-AUPR |
|---|---:|---:|---:|---:|
| self − baseline | -1.22 | -0.89 | +0.40 | +0.99 |
| residual_control − baseline | +0.57 | -0.36 | -0.45 | -0.16 |
| b_near − residual_control | -1.57 | -2.17 | +0.08 | -0.88 |
| c_mid − residual_control | -2.29 | -4.86 | +0.33 | -2.97 |
| d_global − residual_control | -1.81 | -2.86 | -0.13 | -2.15 |
| d_global − c_mid | +0.48 | +2.01 | -0.46 | +0.81 |
| e_shared − d_global | +0.31 | +0.70 | +0.19 | +0.18 |
| memory_context − memory_appearance | +0.83 | +2.00 | +0.41 | +1.10 |
| memory_position − memory_context | -0.58 | -1.16 | -0.02 | -0.57 |
| mosaic_normal − baseline | +1.66 | +4.65 | +2.30 | +2.25 |
| mosaic_normal − mosaic_random | +1.94 | +0.91 | +4.52 | -0.50 |
| mosaic_normal − mosaic_query | +4.67 | +9.87 | +6.48 | +5.67 |

## Interpretation by idea

### 1. Trained multi-range residuals

All near/mid/global residual variants have lower P-AUPR than `residual_control` in both transfer directions. The smallest near-only model also loses, so expansion is not supported by this batch of experiments. Adding global context to the near+mid model recovers some P-AUPR on both datasets but remains below the control.

Sharing weights across ranges (`e_shared`) yields higher P-AUPR than separate weights (`d_global`) on both datasets. This does not support the proposed advantage of decoupling ranges under the current optimization and fusion settings. It is not evidence that contextual residuals can never help.

Important confounds: `b_near` uses only layers 12/15, whereas `c_mid` uses all five layers, so b→c changes both range and layer coverage. Adding branches also increases the summed deep-supervision loss terms without compensating the loss scale. Thus these experiments assess the implemented systems, not a clean isolation of context range or optimization-equivalent architectures. Residual models and the matched control disable CAA mosaic mixing but retain crop CAA.

### 2. Within-image self-similarity

At weight=0.2, exclusion radius=3, k=5, P-AUPR changes by -0.89 pp on MVTec and +0.99 pp on VisA. Keep it as a dataset-dependent option rather than a default.

The MVTec texture-only macro P-AUPR also decreases: 58.04 → 57.37. Grid and leather improve, while tile and wood lose. The proposed broad texture benefit is not supported on this texture subset. DTD was not evaluated.

### 3. Contextual few-shot memory

The joint concatenated key is the clearest positive result. P-AUPR and I-AUROC improve over appearance-only 1-shot memory in all three paired seeds on both datasets. These are equal-shot comparisons; gains over zero-shot also include the effect of adding target normal references.

Restricting matches to a radius of eight patches lowers overall P-AUPR relative to unrestricted contextual matching on both datasets. Some aligned categories benefit, but there is no support for enabling the window globally.

### 4. Inference mosaics

Normal-reference mosaics improve both P-AUPR and I-AUROC over the original baseline on both datasets. Repeating the query is worse than the baseline, so the observed normal-reference gains are not reproduced by query duplication. Normal references beat unverified same-category references in MVTec P-AUPR, but lose to them in VisA P-AUPR; normal references have higher I-AUROC on both datasets. Evidence for normal-reference-specific localization benefits is therefore mixed.

This is a fixed 518×518 canvas with four-corner TTA; each tile has half the linear resolution. The global classifier sees the whole canvas, and the cropped query map supplies the localization peak. The random-reference control samples unverified target test images and is transductive; it is not directly equivalent to an inductive few-shot method. Normal mosaics use one target normal reference despite the stored `shots=0` argument; this argument controls the separate memory bank. Do not describe normal mosaics as zero-shot. The runner does not isolate the cost or effect of four-corner TTA alone.

## Category-level checks

All differences below are mean P-AUPR differences across the same three seeds.

| Test dataset / category | Self − baseline | Context memory − appearance memory | Windowed − unrestricted context memory |
|---|---:|---:|---:|
| MVTEC / transistor | -1.17 | +1.40 | +3.76 |
| MVTEC / cable | +0.10 | +3.18 | +1.61 |
| MVTEC / grid | +3.21 | +0.44 | -0.27 |
| MVTEC / wood | -5.47 | +1.40 | -0.31 |
| MVTEC / screw | -1.76 | +1.75 | -12.52 |
| VISA / pcb1 | -0.65 | +3.15 | -6.76 |
| VISA / pcb2 | +2.19 | +1.58 | -0.28 |
| VISA / pcb3 | -1.14 | +3.04 | +0.17 |
| VISA / pcb4 | -0.48 | +0.14 | +0.18 |

Contextual memory improves all 15 MVTec categories and 9 of 12 VisA categories in mean P-AUPR. The positional window helps transistor and cable but substantially hurts screw and pcb1, consistent with category-dependent alignment assumptions. These category observations are post hoc and should guide follow-up hypotheses, not test-set hyperparameter selection.

## Recommendations and limits

1. Prioritize unrestricted contextual 1-shot memory for the next validation stage. Keep appearance-only memory as the equal-shot control.
2. Pause larger residual models. If revisited, first hold layer coverage and loss scale fixed, then compare relative-magnitude and cosine residuals on auxiliary validation data.
3. Keep self-similarity optional. Its aggregate MVTec and texture-subset results do not justify broader rollout.
4. Continue mosaics only as an exploratory branch: include equal-reference few-shot comparisons, a TTA-only control, and measured inference cost.

Only MVTec and VisA with CLIP are covered. DTD, Real-IAD, KSDD, medical datasets, other backbones, cosine-residual ablations, multiple shot counts, and auxiliary-validation sweeps are absent. Saved configurations record fixed hyperparameters; they do not establish whether a separate auxiliary validation process was performed. Seeds vary trained heads and, for reference-based inference, reference selection; the reported variation combines those effects. No independent timing benchmark or formal significance test is included. Do not pick final hyperparameters from these target test scores.

## Files and regeneration

- [experimental_results.csv](experimental_results.csv): 28 aggregate rows, all seven metrics, standard deviations, paired deltas, short comments, and exact metric-source paths.
- Every CSV score was checked against a fresh macro average of the saved category metrics. Baseline and `mosaic_a` are checked for exact metric equality.

From the repository root:

```bash
.venv/bin/python scripts/summarize_context.py --root runs/context --output reports/context
```
