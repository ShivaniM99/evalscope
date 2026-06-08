# flake8: noqa: E501
"""
livecodebench_pruner.py
=======================
Pure data logic for selecting a representative subset of LiveCodeBench problems
from a set of reference model prediction/review data.

This module has NO evalscope dependencies — it is plain pandas/numpy.
It is imported by livecodebench_pruned_adapter.py which owns the evalscope wiring.

Algorithm (disagreement-first + coverage fill):
  1. Keep ALL problems where any two reference models disagree on pass/fail.
     These are the discriminative problems that separate strong from weak models.
  2. For every (difficulty_tier × problem_category) cell that is still empty
     after step 1, add up to `max_unanimous_fill_per_cell` unanimous problems
     to ensure every cell has at least `min_per_cell` coverage.
     Preference: unanimous all-pass problems first (calibration anchors),
     then any other unanimous problem as a last resort.

Why this generalises to a fourth unknown model:
  The selected indices are fixed from reference data before the new model runs.
  The new model is evaluated on this fixed subset — it cannot influence which
  problems were chosen.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for parsing nested JSON columns
# ──────────────────────────────────────────────────────────────────────────────

def _safe_get(obj: Any, path: Sequence[Any], default=None):
    cur = obj
    for p in path:
        try:
            cur = cur[p] if isinstance(p, int) else cur.get(p)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _maybe_json_load(x: Any) -> Any:
    """Return x as a dict, parsing from JSON string if necessary."""
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return {}
        try:
            return json.loads(x)
        except Exception:
            return {}
    return x if isinstance(x, dict) else {}


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction from raw joined rows
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_model_output(model_output: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (reasoning_text, code_text) from a model_output dict.
    Handles both single-string content and the reasoning+text block format.
    """
    model_output = _maybe_json_load(model_output)
    contents = _safe_get(model_output, ['choices', 0, 'message', 'content'], default=[])

    reasoning_texts: List[str] = []
    code_texts: List[str] = []

    if isinstance(contents, list):
        for item in contents:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type', '')
            text = (
                item.get('reasoning') or
                item.get('text') or
                item.get('content') or
                ''
            )
            if item_type == 'reasoning':
                reasoning_texts.append(text)
            elif item_type == 'text':
                code_texts.append(text)
    elif isinstance(contents, str):
        code_texts.append(contents)

    return '\n'.join(reasoning_texts).strip(), '\n'.join(code_texts).strip()


def _infer_category(model_output: Any) -> str:
    """
    Infer problem category from model reasoning/code text via keyword rules.
    Falls back to 'other'. Used when no metadata category is available.
    """
    reasoning, code = _extract_text_from_model_output(model_output)
    text = (reasoning if reasoning else code).lower()

    rules = [
        (r'dynamic programming|dp\[|dp =|memoiz|knapsack|subproblem|bottom.up|top.down', 'dynamic_programming'),
        (r'\bbfs\b|\bdfs\b|\bdijkstra\b|shortest path|adjacency|topological', 'graph'),
        (r'binary search|bisect|\bmid\s*=|\blo\s*=|\bhi\s*=', 'binary_search_sort'),
        (r'\bprime\b|\bgcd\b|\blcm\b|\bmodulo\b|\bmod\b|factorial|combinat|geometry|arithmetic', 'math'),
        (r'substring|palindrome|prefix|suffix|lexicograph|anagram|string manipulation', 'string'),
        (r'heapq|\bheap\b|\bstack\b|\bdeque\b|priority queue|monotonic', 'data_structure'),
        (r'greedy|locally optimal|always take|always pick', 'greedy'),
        (r'subarray|rotate|sliding window|two pointer|prefix sum', 'array'),
        (r'simulation|second largest|second max|trivial|straightforward|just output|just print|iterate', 'implementation'),
    ]

    for pattern, category in rules:
        if re.search(pattern, text):
            return category
    return 'other'


def _extract_pass(row: Dict[str, Any]) -> float:
    """Extract the binary pass score from a review row."""
    # Try both column name variants the merge may have produced
    raw_score = row.get('sample_score') or row.get('samplescore') or {}
    raw_score = _maybe_json_load(raw_score)

    # Try nested path: {score: {value: {pass: ...}}}
    value = _safe_get(raw_score, ['score', 'value', 'pass'], default=None)
    if value is None:
        value = _safe_get(raw_score, ['value', 'pass'], default=None)
    if value is None:
        value = _safe_get(raw_score, ['pass'], default=None)

    try:
        return float(value)
    except Exception:
        return np.nan


def _extract_output_tokens(row: Dict[str, Any]) -> float:
    model_output = _maybe_json_load(row.get('model_output', {}))
    value = _safe_get(model_output, ['usage', 'output_tokens'], default=None)
    try:
        return float(value)
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Build flat per-row feature table
# ──────────────────────────────────────────────────────────────────────────────

def build_lcb_flat(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a joined prediction+review DataFrame into one row per (index, model)
    with scalar features needed for the pruning algorithm.
    """
    df = joined_df.copy()

    df['pass'] = df.apply(lambda r: _extract_pass(r.to_dict()), axis=1)
    df['problem_category'] = df.apply(
        lambda r: _infer_category(r.get('model_output', {})), axis=1
    )
    df['output_tokens'] = df.apply(lambda r: _extract_output_tokens(r.to_dict()), axis=1)

    keep = ['index', 'model', 'pass', 'problem_category', 'output_tokens']
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Build per-problem feature table
# ──────────────────────────────────────────────────────────────────────────────

def _mode_or_other(series: pd.Series) -> str:
    vals = [x for x in series.dropna().tolist() if x != '']
    if not vals:
        return 'other'
    counts = Counter(vals)
    top_count = max(counts.values())
    top_vals = sorted(k for k, v in counts.items() if v == top_count)
    return top_vals[0] if top_vals else 'other'


def build_problem_feature_table(joined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the flat per-row table into one row per problem (index) with:
      - mean_pass_rate: fraction of reference models that passed
      - model_disagreement: std of pass scores (>0 means models disagree)
      - problem_category: modal inferred category
      - difficulty: 1 - mean_pass_rate
      - difficulty_tier: easy / medium / hard bucketing
      - avg_output_tokens: proxy for problem complexity
    """
    lcb_flat = build_lcb_flat(joined_df)

    per_problem = (
        lcb_flat.groupby('index')
        .agg(
            mean_pass_rate=('pass', 'mean'),
            model_disagreement=('pass', 'std'),
            problem_category=('problem_category', _mode_or_other),
            avg_output_tokens=('output_tokens', 'mean'),
        )
        .reset_index()
    )

    per_problem['model_disagreement'] = per_problem['model_disagreement'].fillna(0.0)
    per_problem['difficulty'] = 1.0 - per_problem['mean_pass_rate']

    def _to_tier(d: float) -> str:
        if d <= 0.33:
            return 'easy'
        if d <= 0.67:
            return 'medium'
        return 'hard'

    per_problem['difficulty_tier'] = per_problem['difficulty'].apply(_to_tier)
    return per_problem


# ──────────────────────────────────────────────────────────────────────────────
# Pruning config + main selection function
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveCodeBenchPruningConfig:
    # K all-pass unanimous problems added per cell for ALL cells (calibration anchors).
    # K=6 is optimal: pearson=1.0, spearman=1.0, mean_abs_delta=0.078 on 3 reference
    # models, selecting 165/315 problems (47% reduction).
    # K=0 gives only discriminative problems (128 problems, higher delta).
    k_allpass_per_cell: int = 6
    # After adding K all-pass fillers, also fill any cell that still has zero
    # coverage using any unanimous problem as a last resort.
    fill_missing_cells: bool = True
    fallback_to_any_unanimous: bool = True


def select_livecodebench_indices(
    joined_df: pd.DataFrame,
    config: Optional[LiveCodeBenchPruningConfig] = None,
) -> List[int]:
    """
    Main entry point. Returns a sorted list of problem indices to keep.

    Algorithm (matches notebook select_indices_with_fixed_k at K=6):

    Step 1 — Keep ALL discriminative problems (model_disagreement > 0).
             These are problems where reference models disagree on pass/fail.
             They carry maximum signal for separating strong from weak models.

    Step 2 — For EVERY (difficulty_tier × category) cell, add up to K
             unanimous all-pass problems as calibration anchors.
             Sort preference: easiest first (lowest difficulty), then shortest
             output (cheapest to evaluate), then lowest index for determinism.
             Rationale: all-pass problems anchor the pass-rate scale so a new
             model's score is comparable across subset sizes.

    Step 3 — (optional) For any cell still empty after steps 1+2, add one
             unanimous problem of any kind as a last-resort coverage fill.
             Sort preference: hardest first (most signal), then longest output.

    Why K=6 is the default:
      Evaluated on 3 reference models, K=6 yields 165/315 problems with
      pearson=1.0, spearman=1.0, mean_abs_delta=0.078, max_abs_delta=0.110.
      K=1 gives 128 problems but mean_abs_delta=0.176 — worse calibration.
      K=8 gives 177 problems with diminishing returns on delta.

    Why this generalises to a fourth unknown model:
      The index set is fixed from reference data before any new model runs.
      The new model cannot influence which problems are selected.
    """
    config = config or LiveCodeBenchPruningConfig()
    per_problem = build_problem_feature_table(joined_df)

    discriminative = per_problem[per_problem['model_disagreement'] > 0].copy()
    unanimous      = per_problem[per_problem['model_disagreement'] == 0].copy()
    unanimous_allpass = unanimous[unanimous['mean_pass_rate'] == 1.0].copy()

    # Step 1: all discriminative problems
    selected = set(discriminative['index'].tolist())

    all_cells = sorted(
        set(zip(
            per_problem['difficulty_tier'].astype(str),
            per_problem['problem_category'].astype(str),
        ))
    )

    # Step 2: add K all-pass unanimous per cell for ALL cells
    # Sort: easiest first → shortest output → lowest index (deterministic)
    unanimous_allpass_sorted = unanimous_allpass.sort_values(
        ['difficulty', 'avg_output_tokens', 'index'],
        ascending=[True, True, True],
    )

    for tier, category in all_cells:
        group = unanimous_allpass_sorted[
            (unanimous_allpass_sorted['difficulty_tier'].astype(str) == tier) &
            (unanimous_allpass_sorted['problem_category'].astype(str) == category)
        ]
        if len(group) == 0:
            continue
        take_n = min(config.k_allpass_per_cell, len(group))
        selected.update(group.head(take_n)['index'].tolist())

    # Step 3: fill any cell that still has zero coverage
    if config.fill_missing_cells and config.fallback_to_any_unanimous:
        selected_df = per_problem[per_problem['index'].isin(selected)]
        covered_cells = set(zip(
            selected_df['difficulty_tier'].astype(str),
            selected_df['problem_category'].astype(str),
        ))
        missing_cells = [c for c in all_cells if c not in covered_cells]

        # Sort: hardest first → longest output → lowest index
        unanimous_sorted = unanimous.sort_values(
            ['difficulty', 'avg_output_tokens', 'index'],
            ascending=[False, False, True],
        )

        for tier, category in missing_cells:
            group = unanimous_sorted[
                (unanimous_sorted['difficulty_tier'].astype(str) == tier) &
                (unanimous_sorted['problem_category'].astype(str) == category) &
                (~unanimous_sorted['index'].isin(selected))
            ]
            if len(group) == 0:
                continue
            selected.add(int(group.iloc[0]['index']))

    return sorted(int(x) for x in selected)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: filter records list by selected indices
# ──────────────────────────────────────────────────────────────────────────────

def filter_records_by_indices(
    records: Iterable[Dict[str, Any]],
    selected_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    """Filter a list of raw dataset records to only those in selected_indices."""
    selected = set(int(x) for x in selected_indices)
    out = []
    for record in records:
        idx = record.get('index')
        if idx is None:
            continue
        try:
            if int(idx) in selected:
                out.append(record)
        except (TypeError, ValueError):
            continue
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Utility: load joined reference DataFrame from disk
# ──────────────────────────────────────────────────────────────────────────────

def load_joined_reference_dataframe(
    joined_csv_path: Optional[str] = None,
    prediction_dir: Optional[str] = None,
    review_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load reference model predictions + reviews into a single joined DataFrame.

    Provide either:
      - joined_csv_path: a pre-built CSV (fastest, recommended for production)
      - prediction_dir + review_dir: directories of JSONL files (auto-joined on index+model)
    """
    if joined_csv_path:
        path = Path(joined_csv_path)
        if not path.exists():
            raise FileNotFoundError(f'joined_csv_path not found: {joined_csv_path}')
        return pd.read_csv(path)

    if not prediction_dir or not review_dir:
        raise ValueError(
            'Provide either joined_csv_path or both prediction_dir and review_dir.'
        )

    pred_dir = Path(prediction_dir)
    rev_dir = Path(review_dir)

    if not pred_dir.exists():
        raise FileNotFoundError(f'prediction_dir not found: {prediction_dir}')
    if not rev_dir.exists():
        raise FileNotFoundError(f'review_dir not found: {review_dir}')

    def _load_jsonl(path: Path) -> pd.DataFrame:
        rows = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def _load_dir(directory: Path, glob: str, stem_strip: str) -> pd.DataFrame:
        dfs = []
        for p in directory.glob(glob):
            df = _load_jsonl(p)
            if 'model' not in df.columns:
                # Derive model name from filename by stripping known prefixes
                model_name = p.stem
                for prefix in [stem_strip, 'live_code_bench_v5__', 'live_code_bench_', 'livecodebench']:
                    model_name = model_name.replace(prefix, '')
                df['model'] = model_name.strip('-_')
            dfs.append(df)
        if not dfs:
            raise RuntimeError(f'No files matching {glob!r} found in {directory}')
        return pd.concat(dfs, ignore_index=True)

    # Accept both naming conventions seen in the task data
    pred_all = _load_dir(pred_dir, 'live_code_bench_*.jsonl', 'live_code_bench_v5__')
    rev_all  = _load_dir(rev_dir,  'live_code_bench_*.jsonl', 'live_code_bench_v5__')

    join_keys = (
        ['index', 'model']
        if ('model' in pred_all.columns and 'model' in rev_all.columns)
        else ['index']
    )

    joined = pred_all.merge(rev_all, on=join_keys, how='inner', suffixes=('_pred', '_rev'))

    # Normalise score column name
    if 'sample_score' not in joined.columns and 'samplescore' in joined.columns:
        joined['sample_score'] = joined['samplescore']

    return joined
