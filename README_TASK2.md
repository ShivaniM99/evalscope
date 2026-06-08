# Task 2 — Benchmark Compression: LiveCodeBench Pruner

This documents the Task 2 Part 1 implementation: a pruned LiveCodeBench adapter
living inside this `evalscope` fork.

---

## Evalscope commit SHA

Pin this before running:

```bash
git -C /path/to/this/repo rev-parse HEAD
```

> Developed against the evalscope commit on this branch. The framework APIs are
> still evolving — pin this SHA when reviewing.

---

## What was implemented

### New files

| File | Purpose |
|---|---|
| `evalscope/benchmarks/live_code_bench/pruners/livecodebench_pruner.py` | Pure-data pruning algorithm (no evalscope deps) |
| `evalscope/benchmarks/live_code_bench/livecodebench_pruned_adapter.py` | evalscope adapter — registers `live_code_bench_pruned` |
| `evalscope_ext/tools/compare_runs.py` | CLI tool to compare full vs pruned run output |

### Modified files

| File | Change |
|---|---|
| `evalscope/benchmarks/live_code_bench/live_code_bench_adapter.py` | `__init__` now reads `start_date`/`end_date` from `extra_params` so `sample_filter` works correctly |

---

## Pruning algorithm

**Disagreement-first + calibration fill** (implemented in `livecodebench_pruner.py`):

1. **Keep all discriminative problems** — any problem where at least one pair of
   the 3 reference models disagrees on pass/fail. These carry maximum signal for
   separating strong from weak models. (128 problems out of 315.)

2. **Add K all-pass calibration anchors per cell** — for every
   `(difficulty_tier × problem_category)` cell, add up to K unanimous all-pass
   problems sorted by easiest-first, then shortest output, then lowest index.
   These anchor the pass-rate scale so a new model's score is comparable.

3. **Fill missing cells** — if any cell still has zero coverage after step 2,
   add any unanimous problem from that cell as a last resort.

**Why K=6 is the default:**
Evaluated across K = 0–6 on the 3 reference models. K=6 achieves Pearson=1.0,
Spearman=1.0, mean absolute delta=0.078 while selecting **165/315 problems
(47% reduction)**. K=0 (discriminative only) gives 128 problems but higher
score delta.

**Why this generalises to a fourth model:**
The index set is computed once from reference data and is fixed before the new
model runs. The new model cannot influence which problems are selected.

---

## Install

```bash
git clone <your-private-repo-url>
cd evalscope
pip install -e ".[all]"
pip install scipy          # required by compare_runs
```

---

## Run contract

### Step 1 — Full benchmark (baseline)

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets live_code_bench \
  --output ./results_full/
```

### Step 2 — Pruned benchmark

Pass reference data paths so the pruner can build the index set.
`prediction_dir` and `review_dir` point to the `Evals/Part 1/` directories
from the challenge repo.

```bash
evalscope eval \
  --model <model_name_or_path> \
  --datasets live_code_bench_pruned \
  --dataset-args '{
      "prediction_dir": "/path/to/Evals/Part 1/predictions",
      "review_dir":     "/path/to/Evals/Part 1/reviews",
      "k_allpass_per_cell": 6,
      "fill_missing_cells": true
  }' \
  --output ./results_pruned/
```

**Extra params reference:**

| Param | Default | Description |
|---|---|---|
| `prediction_dir` | null | Path to reference prediction JSONL files |
| `review_dir` | null | Path to reference review JSONL files |
| `joined_csv_path` | null | Pre-built joined CSV (faster alternative to prediction_dir+review_dir) |
| `k_allpass_per_cell` | 6 | All-pass calibration anchors per (tier × category) cell |
| `fill_missing_cells` | true | Fill zero-coverage cells with any unanimous problem |
| `start_date` | null | Filter problems from this date (YYYY-MM-DD) |
| `end_date` | null | Filter problems up to this date (YYYY-MM-DD) |

### Step 3 — Compare runs

```bash
python -m evalscope_ext.tools.compare_runs \
  --full  ./results_full/ \
  --pruned ./results_pruned/
```

This prints per-model pass rates, rank comparison, and Spearman ρ between
full and pruned rankings.

---

## Verifying the pipeline without a live model

The notebook `Task2/LiveCodeBench/LiveCodeBench_exploration.ipynb` in the
challenge repo runs the pruner end-to-end against the 3 reference models and
verifies all assertions (index bounds, no duplicates, full cell coverage,
rank preservation). Run cells 46–55 to reproduce.

---

## Key results on reference data (3 models)

| K | Problems selected | Reduction | Pearson | Spearman | Mean |Δ| pass rate |
|---|---|---|---|---|---|
| 0 | 128 / 315 | 59% | 1.0 | 1.0 | higher |
| 6 | 165 / 315 | 47% | 1.0 | 1.0 | 0.078 |

K=6 is the default. It preserves perfect rank ordering across all 3 reference
models while cutting nearly half the benchmark.
