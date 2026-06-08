# flake8: noqa: E501
"""
livecodebench_pruned_adapter.py
================================
evalscope adapter for the PRUNED LiveCodeBench variant.

Pruning is wired via PrunedAdapterMixin — the universal base that handles
the evalscope sample_filter hook and lazy index caching.  This class only
needs to implement _select_indices() with LCB-specific logic.

Algorithm
---------
1. Keep ALL problems where any pair of reference models disagrees on pass/fail.
2. Add K unanimous all-pass problems per (difficulty × category) cell as
   calibration anchors.  K=6 → 165/315 problems, Pearson=1.0, Spearman=1.0.
3. Fill any cell that still has zero coverage with any unanimous problem.

Parameters (extra_params)
-------------------------
prediction_dir      : dir of reference prediction JSONL files
review_dir          : dir of reference review JSONL files
joined_csv_path     : pre-built joined CSV (faster alternative)
k_allpass_per_cell  : int  [default 6]
fill_missing_cells  : bool [default True]
start_date          : YYYY-MM-DD or null
end_date            : YYYY-MM-DD or null
"""
from typing import List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope.benchmarks._pruned_mixin import PrunedAdapterMixin
from .live_code_bench_adapter import LiveCodeBenchAdapter
from .pruners.livecodebench_pruner import (
    LiveCodeBenchPruningConfig,
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

### Selection strategy: disagreement-first + calibration fill

1. **Keep all discriminative problems** — any problem where at least one pair of
   reference models disagree on pass/fail.

2. **Add K all-pass calibration anchors per (difficulty × category) cell** — ensures
   every cell has baseline coverage and anchors the pass-rate scale.

3. **Fill missing cells** — last-resort unanimous filler for zero-coverage cells.

### Why this works for an unknown fourth model

The index set is computed once from reference data and is **fixed** before the
new model runs.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `prediction_dir` | null | Directory of reference prediction JSONL files |
| `review_dir` | null | Directory of reference review JSONL files |
| `joined_csv_path` | null | Pre-built joined CSV (fastest option) |
| `k_allpass_per_cell` | 6 | All-pass anchors per cell (K=6 → 165/315, Pearson=1.0) |
| `fill_missing_cells` | true | Fill zero-coverage cells with any unanimous problem |
| `start_date` | null | Filter problems from this date (YYYY-MM-DD) |
| `end_date` | null | Filter problems up to this date (YYYY-MM-DD) |
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
                'description': 'Filter problems from this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'joined_csv_path': {
                'type': 'str | null',
                'description': 'Path to pre-joined prediction/review CSV.',
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
                'description': 'All-pass unanimous problems per cell. K=6 is optimal.',
                'value': 6,
            },
            'fill_missing_cells': {
                'type': 'bool',
                'description': 'Fill zero-coverage cells with any unanimous problem.',
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
class LiveCodeBenchPrunedAdapter(PrunedAdapterMixin, LiveCodeBenchAdapter):
    """
    LiveCodeBench adapter that evaluates only a pruned subset of problems.

    Inherits universal pruning plumbing from PrunedAdapterMixin and
    implements _select_indices() with LCB-specific disagreement-first logic.
    """

    def __init__(self, benchmark_meta: BenchmarkMeta, task_config=None):
        super().__init__(benchmark_meta, task_config)
        self.__init_pruned_mixin__()

        self.joined_csv_path: Optional[str] = self._get_extra_param('joined_csv_path')
        self.prediction_dir:  Optional[str] = self._get_extra_param('prediction_dir')
        self.review_dir:      Optional[str] = self._get_extra_param('review_dir')
        self.k_allpass_per_cell: int        = int(self._get_extra_param('k_allpass_per_cell', 6))
        self.fill_missing_cells: bool       = bool(self._get_extra_param('fill_missing_cells', True))

    # ── benchmark-specific selection logic ────────────────────────────────────

    def _select_indices(self) -> List[int]:
        joined_df = load_joined_reference_dataframe(
            joined_csv_path=self.joined_csv_path,
            prediction_dir=self.prediction_dir,
            review_dir=self.review_dir,
        )
        config = LiveCodeBenchPruningConfig(
            k_allpass_per_cell=self.k_allpass_per_cell,
            fill_missing_cells=self.fill_missing_cells,
        )
        indices = select_livecodebench_indices(joined_df, config=config)
        logger.info(
            f'LiveCodeBenchPrunedAdapter: selected {len(indices)} problems '
            f'(range {min(indices)}–{max(indices)}).'
        )
        return indices

    # ── evalscope hook ────────────────────────────────────────────────────────

    def sample_filter(self, sample: Sample) -> bool:
        """Apply parent date filter first, then pruned index filter."""
        if not LiveCodeBenchAdapter.sample_filter(self, sample):
            return False
        return PrunedAdapterMixin.sample_filter(self, sample)
