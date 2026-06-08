# Task 2 — Benchmark Compression: LiveCodeBench + AA-LCR + MMMU Pruners

This documents the Task 2 implementation: pruned adapters for LiveCodeBench,
AA-LCR, and MMMU living inside this `evalscope` fork.

---

## Evalscope commit SHA

```
2a1de79f60b8d719e2ffb121aab6c21264aeffa7
```

> Developed against this evalscope commit. Pin this SHA when reviewing.

---

## Architecture

All three pruned benchmarks share one base class — `PrunedAdapterMixin`
(`evalscope/benchmarks/_pruned_mixin.py`). It handles the evalscope
`sample_filter` hook, lazy key caching, and the `_get_extra_param` helper.
Each benchmark adapter only needs to implement `_select_indices()` with its
own algorithm. This follows the same separate-adapter pattern as
`live_code_bench_pruned`.

---

## Pruning algorithms

### LiveCodeBench — Disagreement-first + calibration fill

1. **Keep all discriminative problems** — any problem where at least one pair of
   the 3 reference models disagrees on pass/fail. (128/315 problems.)
2. **Add K all-pass calibration anchors per cell** — for every
   `(difficulty_tier × problem_category)` cell, add up to K unanimous all-pass
   problems to anchor the pass-rate scale.
3. **Fill missing cells** — if any cell still has zero coverage, add any
   unanimous problem as a last resort.

K=6 default: 165/315 problems, Pearson=1.0, Spearman=1.0, mean |Δ|=0.078.

---

### AA-LCR — Stratified discriminability pruning

1. **Keep all discriminative questions** — any question with `acc_range > 0`
   across the 3 reference models.
2. **Stratified sample from non-discriminative pool** — bucketed by
   (difficulty × reasoning-complexity) quadrant.

Quadrant weights:

| Quadrant | Retention |
|---|---|
| hard + complex | 100% |
| hard + simple | 25% |
| easy + complex | 100% |
| easy + simple | 5% |

---

### MMMU — Image-type encoder-stress pruning

Selects questions that stress the **vision encoder** specifically, so model
quality scores reflect encoder capability rather than language-head priors.

1. **HIGH-stress samples** (all kept) — medical scans, chemical structures,
   geometric shapes, technical blueprints, microscopy. Cannot be answered
   correctly without fine-grained visual perception.
2. **MEDIUM-stress samples** (stratified sample by difficulty) — diagrams,
   charts, maps, sketches.
3. **LOW-stress calibration anchors** — Hard-difficulty only; kept as baseline.

MMMU note: integer `index` is per-subset and not globally unique. The unique
key is the string `id` (e.g. `validation_Accounting_23`). The mixin's
`_get_sample_key()` is overridden to use this string ID.

---

## Install

Clone and install the **evalscope fork** (this is where all the pruned adapter
code lives):

```bash
git clone <your-evalscope-fork-url>
cd evalscope
pip install -e ".[all]"
pip install scipy          # required by compare_runs
```

The **challenge repo** (`ai-model-quality-challenge-main`) is only needed for
the reference model files in `Evals/`. If you already have it cloned locally,
just set `CHALLENGE_REPO` to point at it (see Run contracts below).

---

## Run contracts

> **If you cloned the challenge repo**, set this variable first and use it in
> all commands below — no path editing needed:
>
> ```bash
> export CHALLENGE_REPO=/path/to/ai-model-quality-challenge-main
> ```

---

### LiveCodeBench

#### Full benchmark (baseline)

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets live_code_bench \
  --output ./results_lcb_full/
```

#### Pruned benchmark

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets live_code_bench_pruned \
  --dataset-args "{
      \"prediction_dir\": \"$CHALLENGE_REPO/Evals/Part 1/predictions\",
      \"review_dir\":     \"$CHALLENGE_REPO/Evals/Part 1/reviews\",
      \"k_allpass_per_cell\": 6,
      \"fill_missing_cells\": true
  }" \
  --output ./results_lcb_pruned/
```

Extra params:

| Param | Default | Description |
|---|---|---|
| `prediction_dir` | null | Reference prediction JSONL directory |
| `review_dir` | null | Reference review JSONL directory |
| `joined_csv_path` | null | Pre-built joined CSV (faster alternative) |
| `k_allpass_per_cell` | 6 | All-pass anchors per cell |
| `fill_missing_cells` | true | Fill zero-coverage cells |
| `start_date` | null | Filter from date (YYYY-MM-DD) |
| `end_date` | null | Filter to date (YYYY-MM-DD) |

---

### AA-LCR

#### Full benchmark (baseline)

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets aa_lcr \
  --output ./results_aalcr_full/
```

#### Pruned benchmark

The adapter builds the flat scores table automatically from the raw reference
model files — no pre-built CSV needed.

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets aa_lcr_pruned \
  --dataset-args "{
      \"predictions_dir\": \"$CHALLENGE_REPO/Evals/Part 1/predictions\",
      \"reviews_dir\":     \"$CHALLENGE_REPO/Evals/Part 1/reviews\",
      \"random_state\": 42
  }" \
  --output ./results_aalcr_pruned/
```

Extra params:

| Param | Default | Description |
|---|---|---|
| `predictions_dir` | null | Dir of reference prediction JSONL files (`aa_lcr__{model}.jsonl`) |
| `reviews_dir` | null | Dir of reference review JSONL files (`aa_lcr__{model}.jsonl`) |
| `scores_path` | null | Alternative: pre-built flat CSV/JSONL (columns: `index, model, acc, reasoning_len, input_tokens`) |
| `random_state` | 42 | Reproducibility seed |
| `text_dir` | null | Local AA-LCR documents dir; auto-downloaded if omitted |

---

### MMMU

#### Full benchmark (baseline)

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets mmmu \
  --output ./results_mmmu_full/
```

#### Pruned benchmark — with reference prediction files (recommended)

Pass the challenge repo's MMMU prediction directory to load image-type metadata
without re-downloading the full dataset:

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets mmmu_pruned \
  --dataset-args "{
      \"predictions_dir\": \"$CHALLENGE_REPO/Evals/MMMU/predictions\",
      \"medium_sample_frac\": 0.5,
      \"keep_hard_low\": true,
      \"random_state\": 42
  }" \
  --output ./results_mmmu_pruned/
```

#### Pruned benchmark — without prior predictions

If no prior predictions exist, omit `predictions_dir`. The adapter will load
metadata directly from the HuggingFace dataset:

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets mmmu_pruned \
  --dataset-args '{
      "medium_sample_frac": 0.5,
      "keep_hard_low": true,
      "random_state": 42
  }' \
  --output ./results_mmmu_pruned/
```

Extra params:

| Param | Default | Description |
|---|---|---|
| `predictions_dir` | null | Dir of MMMU prediction JSONL files for image-type metadata loading |
| `medium_sample_frac` | 0.5 | Fraction of medium-stress samples per difficulty stratum |
| `keep_hard_low` | true | Keep Hard low-stress samples as calibration anchors |
| `random_state` | 42 | Reproducibility seed |

---

## Compare full vs pruned runs

```bash
python -m evalscope_ext.tools.compare_runs \
  --full  ./results_<benchmark>_full/ \
  --pruned ./results_<benchmark>_pruned/
```

Prints per-model pass rates, rank comparison, and Spearman ρ between
full and pruned rankings.

---

## Verifying the pipeline (no live model needed)

The notebook `Task2/LiveCodeBench/LiveCodeBench_exploration.ipynb` in the
challenge repo runs the LCB pruner end-to-end against the 3 reference models
and verifies all assertions (index bounds, no duplicates, full cell coverage,
rank preservation).

---

## Key results on reference data (3 models)

### LiveCodeBench

| K | Problems selected | Reduction | Pearson | Spearman | Mean |Δ| pass rate |
|---|---|---|---|---|---|---|
| 0 | 128 / 315 | 59% | 1.0 | 1.0 | higher |
| 6 | 165 / 315 | 47% | 1.0 | 1.0 | 0.078 |

Verified output:

```
✅ Adapter _pruned_indices populated: 165 problems
✅ Index range: 4 – 312
✅ Every kept sample's index is in _pruned_indices
✅ First 20 selected: [4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18, 19, 20, 21, 24, 27, 28, 31, 33, 40]
✅ All assertions passed
```

---

### AA-LCR

| Metric | Value |
|---|---|
| Questions selected | 57 / 100 |
| Reduction | 43% |
| Discriminative questions retained | 43 / 43 (100%) |

Verified output:

```
✅ Pruned set : 57 / 100 questions
✅ Reduction  : 43.0%
✅ All 43 discriminative questions retained
✅ All assertions passed
```

---

### MMMU

The image-type pruner focuses on questions where encoder quality is the
primary determinant of correctness. The selection is model-agnostic and
reproducible from `random_state` alone — no reference model scores needed.
