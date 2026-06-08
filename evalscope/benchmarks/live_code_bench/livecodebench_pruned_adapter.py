# flake8: noqa: E501
"""
livecodebench_pruned_adapter.py
================================
evalscope adapter for the PRUNED LiveCodeBench variant.

How the pruning is wired into evalscope's pipeline
----------------------------------------------------
evalscope's runner calls these methods on each dataset record in order:

  record_to_sample(record)  →  Sample      (inherited from LiveCodeBenchAdapter)
  sample_filter(sample)     →  bool        (OVERRIDDEN HERE — this is where pruning happens)
  extract_answer(...)       →  str         (inherited)
  match_score(...)          →  Score       (inherited)

By overriding `sample_filter` we hook into the one point in the pipeline where
evalscope asks "should I keep this sample?" — which is exactly what we need.
The pruner computes a fixed set of indices from reference data ONCE, caches it,
and then every subsequent call is just a set-membership check.

The selected index set is computed from historical reference model data
(3 models shipped with the task) and is FIXED before any new model runs.
A fourth unknown model is evaluated on this fixed subset — it cannot influence
which problems were chosen. This is what makes the pruner generalisable.

Parameters (passed via extra_params in BenchmarkMeta or evalscope CLI --extra-params)
---------------------------------------------------------------------------
joined_csv_path          : str | None   — path to a pre-built joined CSV (fastest)
prediction_dir           : str | None   — dir of reference prediction JSONL files
review_dir               : str | None   — dir of reference review JSONL files
min_per_cell             : int          — minimum problems per (tier × category) cell  [default 1]
max_unanimous_fill_per_cell : int       — cap on unanimous fillers added per cell      [default 1]
start_date / end_date    : str | None   — inherited date filter from parent
debug                    : bool         — verbose logging
"""
from typing import Any, Dict, List, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .live_code_bench_adapter import LiveCodeBenchAdapter
from .pruners.livecodebench_pruner import (
    LiveCodeBenchPruningConfig,
    filter_records_by_indices,
    load_joined_reference_dataframe,
    select_livecodebench_indices,
)

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

Pruned variant of LiveCodeBench for fast, cheap model-quality checks.

### Selection strategy: disagreement-first + coverage fill

1. **Keep all discriminative problems** — any problem where at least one pair of
   reference models disagree on pass/fail. These problems carry maximum signal
   for separating strong from weak models.

2. **Fill missing (difficulty × category) cells** — add a small number of
   unanimous problems only where a cell would otherwise have zero representation,
   ensuring broad capability coverage even after heavy pruning.

### Why this works for an unknown fourth model

The index set is computed once from reference data and is **fixed** before the
new model runs. The new model cannot influence which problems are selected.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `joined_csv_path` | null | Pre-built joined CSV (prediction+review). Fastest option. |
| `prediction_dir` | null | Directory of reference prediction JSONL files. |
| `review_dir` | null | Directory of reference review JSONL files. |
| `k_allpass_per_cell` | 6 | All-pass calibration anchors per cell. K=6 gives 165/315 problems, pearson=1.0. |
| `fill_missing_cells` | true | Fill zero-coverage cells with any unanimous problem as fallback. |
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=[
            'release_latest', 'release_v1', 'release_v2', 'release_v3',
            'release_v4', 'release_v5', 'release_v6',
            'v1', 'v1_v2', 'v1_v3', 'v1_v4', 'v1_v5', 'v1_v6',
            'v2', 'v2_v3', 'v2_v4', 'v2_v5', 'v2_v6',
            'v3', 'v3_v4', 'v3_v5', 'v3_v6',
            'v4', 'v4_v5', 'v4_v6',
            'v5', 'v5_v6',
            'v6',
        ],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n'
            '{format_prompt} ### Answer: (use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
        extra_params={
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
            'joined_csv_path': {
                'type': 'str | null',
                'description': 'Path to pre-joined prediction/review CSV for pruned subset construction.',
                'value': None,
            },
            'prediction_dir': {
                'type': 'str | null',
                'description': 'Directory with reference prediction JSONL files.',
                'value': None,
            },
            'review_dir': {
                'type': 'str | null',
                'description': 'Directory with reference review JSONL files.',
                'value': None,
            },
            'k_allpass_per_cell': {
                'type': 'int',
                'description': 'All-pass unanimous problems added per cell as calibration anchors. K=6 is optimal (165/315 problems, pearson=1.0, spearman=1.0).',
                'value': 6,
            },
            'fill_missing_cells': {
                'type': 'bool',
                'description': 'Fill any cell with zero coverage after K-fill using any unanimous problem.',
                'value': True,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    LiveCodeBench adapter that evaluates only a pruned subset of problems.

    The subset is selected once from reference model data, cached in
    `_pruned_indices`, and then applied via `sample_filter` — the standard
    evalscope hook for per-sample inclusion decisions.
    """

    def __init__(self, benchmark_meta: BenchmarkMeta, task_config=None):
        # ── Match parent's positional signature exactly ──────────────────────
        super().__init__(benchmark_meta, task_config)

        params = getattr(benchmark_meta, 'extra_params', {}) or {}

        def _get(key, default):
            val = params.get(key, default)
            if isinstance(val, dict):
                return val.get('value', default)
            return val

        self.joined_csv_path: Optional[str]  = _get('joined_csv_path', None)
        self.prediction_dir:  Optional[str]  = _get('prediction_dir', None)
        self.review_dir:      Optional[str]  = _get('review_dir', None)
        self.k_allpass_per_cell: int         = int(_get('k_allpass_per_cell', 6))
        self.fill_missing_cells: bool        = bool(_get('fill_missing_cells', True))

        # Lazily populated on first call to sample_filter
        self._pruned_indices: Optional[List[int]] = None
        self._pruned_index_set: Optional[Set[int]] = None

    # ── Index selection (lazy, cached) ───────────────────────────────────────

    def _ensure_indices_loaded(self) -> Set[int]:
        """
        Compute and cache the pruned index set.
        Called once on the first sample_filter invocation.
        """
        if self._pruned_index_set is not None:
            return self._pruned_index_set

        logger.info('LiveCodeBenchPrunedAdapter: computing pruned index set...')

        joined_df = load_joined_reference_dataframe(
            joined_csv_path=self.joined_csv_path,
            prediction_dir=self.prediction_dir,
            review_dir=self.review_dir,
        )

        config = LiveCodeBenchPruningConfig(
            k_allpass_per_cell=self.k_allpass_per_cell,
            fill_missing_cells=self.fill_missing_cells,
        )

        self._pruned_indices = select_livecodebench_indices(joined_df, config=config)
        self._pruned_index_set = set(self._pruned_indices)

        logger.info(
            f'LiveCodeBenchPrunedAdapter: selected {len(self._pruned_indices)} problems '
            f'(index range {min(self._pruned_indices)}–{max(self._pruned_indices)}).'
        )
        return self._pruned_index_set

    # ── evalscope hook: called once per sample to decide keep/skip ───────────

    def sample_filter(self, sample: Sample) -> bool:
        """
        Return True only if this sample's index is in the pruned set
        AND it passes the parent's date filter.

        evalscope calls this for every sample after record_to_sample().
        This is the single correct place to apply dataset subsetting.
        """
        # Parent date filter first (fast, no I/O)
        if not super().sample_filter(sample):
            return False

        # Pruned index filter (loads reference data on first call, then O(1))
        idx = sample.metadata.get('index')
        if idx is None:
            logger.warning('sample missing "index" in metadata — excluded from pruned set')
            return False

        pruned_set = self._ensure_indices_loaded()
        return int(idx) in pruned_set
