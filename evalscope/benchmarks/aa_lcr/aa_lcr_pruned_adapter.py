# flake8: noqa: E501
"""
aa_lcr_pruned_adapter.py
=========================
evalscope adapter for the PRUNED AA-LCR variant.

Pruning is wired via PrunedAdapterMixin — the universal base that handles
the evalscope sample_filter hook and lazy index caching.  This class only
needs to implement _select_indices() with AA-LCR-specific logic.

Algorithm — Stratified Discriminability Pruner
----------------------------------------------
Stage 1: Keep ALL discriminative questions (acc_range > 0 across reference models).
         These carry maximum signal for separating strong from weak models.

Stage 2: Stratified sample of non-discriminative questions by
         (difficulty × reasoning-complexity) quadrant.
         Hard/complex questions are prioritised; easy/simple are thinned out.

         Quadrant weights (tunable):
           hard + complex  → 100%
           hard + simple   →  25%
           easy + complex  → 100%
           easy + simple   →   5%

Why this generalises to a fourth model
---------------------------------------
The index set is fixed from reference data before the new model runs.
Model 4 never contributes to index selection — it is simply evaluated on
the already-fixed subset.

Parameters (extra_params)
--------------------------
predictions_dir : path to dir of reference model prediction JSONL files
                  (named aa_lcr__{model}.jsonl — from Evals/Part 1/predictions)
reviews_dir     : path to dir of reference model review JSONL files
                  (named aa_lcr__{model}.jsonl — from Evals/Part 1/reviews)
scores_path     : alternative to predictions_dir+reviews_dir — path to a
                  pre-built flat CSV/JSONL (columns: index, model, acc,
                  reasoning_len, input_tokens).
random_state    : int [default 42]
text_dir        : inherited from base — local AA-LCR document directory
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope.benchmarks._pruned_mixin import PrunedAdapterMixin
from .aa_lcr_adapter import AALCRAdapter
from .pruner.stratified_pruner import StratifiedDiscriminabilityPruner

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

Pruned variant of AA-LCR for fast, cheap long-context capability checks.

### Selection strategy: stratified discriminability

1. **Keep all discriminative questions** — any question where reference models
   disagree (acc_range > 0). Maximum signal for separating strong from weak models.

2. **Stratified sample of non-discriminative questions** — bucketed by
   (difficulty × reasoning-complexity) quadrant. Hard/complex questions are
   retained at 100%; easy/simple at 5%.

### Why this works for an unknown fourth model

The index set is fixed from reference data before the new model runs.
Model 4 never contributes to index selection.

### Input: reference model data

Point `predictions_dir` and `reviews_dir` at the challenge repo's
`Evals/Part 1/predictions` and `Evals/Part 1/reviews` directories.
The adapter automatically builds the flat scores table from those files.

Alternatively supply `scores_path` if you have a pre-built flat CSV/JSONL.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `predictions_dir` | null | Dir of reference prediction JSONL files (aa_lcr__{model}.jsonl) |
| `reviews_dir` | null | Dir of reference review JSONL files (aa_lcr__{model}.jsonl) |
| `scores_path` | null | Pre-built flat CSV/JSONL (alternative to predictions_dir+reviews_dir) |
| `random_state` | 42 | Reproducibility seed |
| `text_dir` | null | Local AA-LCR documents dir (auto-downloaded if omitted) |
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory with extracted AA-LCR text files; auto-downloaded if null.',
                'value': None,
            },
            'predictions_dir': {
                'type': 'str | null',
                'description': (
                    'Directory of reference model prediction JSONL files '
                    '(aa_lcr__{model}.jsonl). From Evals/Part 1/predictions in the challenge repo.'
                ),
                'value': None,
            },
            'reviews_dir': {
                'type': 'str | null',
                'description': (
                    'Directory of reference model review JSONL files '
                    '(aa_lcr__{model}.jsonl). From Evals/Part 1/reviews in the challenge repo.'
                ),
                'value': None,
            },
            'scores_path': {
                'type': 'str | null',
                'description': (
                    'Alternative to predictions_dir+reviews_dir. '
                    'Path to pre-built flat CSV or JSONL with columns: '
                    'index, model, acc, reasoning_len, input_tokens.'
                ),
                'value': None,
            },
            'random_state': {
                'type': 'int',
                'description': 'Random seed for stratified sampler reproducibility.',
                'value': 42,
            },
        },
    )
)
class AALCRPrunedAdapter(PrunedAdapterMixin, AALCRAdapter):
    """
    AA-LCR adapter that evaluates only a pruned subset of questions.

    Inherits universal pruning plumbing from PrunedAdapterMixin and
    implements _select_indices() with AA-LCR stratified discriminability logic.

    Accepts either:
      - predictions_dir + reviews_dir  →  flat scores table built automatically
      - scores_path                    →  pre-built flat CSV/JSONL
    """

    def __init__(self, benchmark_meta: BenchmarkMeta, task_config=None):
        super().__init__(benchmark_meta, task_config)
        self.__init_pruned_mixin__()

        self.predictions_dir: Optional[str] = self._get_extra_param('predictions_dir')
        self.reviews_dir:     Optional[str] = self._get_extra_param('reviews_dir')
        self.scores_path:     Optional[str] = self._get_extra_param('scores_path')
        self.random_state:    int           = int(self._get_extra_param('random_state', 42))

    # ── benchmark-specific selection logic ────────────────────────────────────

    def _select_indices(self) -> List[int]:
        scores_df = self._load_scores()
        pruner = StratifiedDiscriminabilityPruner(random_state=self.random_state)
        indices = pruner.fit_transform(scores_df)
        logger.info(
            f'AALCRPrunedAdapter: selected {len(indices)}/100 questions '
            f'(random_state={self.random_state}).'
        )
        return indices

    def _load_scores(self) -> pd.DataFrame:
        """
        Build the flat scores DataFrame from whichever source was provided.

        Priority: predictions_dir+reviews_dir > scores_path
        """
        if self.predictions_dir and self.reviews_dir:
            logger.info(
                f'AALCRPrunedAdapter: building scores from '
                f'{self.predictions_dir} + {self.reviews_dir}'
            )
            return StratifiedDiscriminabilityPruner.build_scores_df(
                self.predictions_dir, self.reviews_dir
            )

        if self.scores_path:
            p = Path(self.scores_path)
            logger.info(f'AALCRPrunedAdapter: loading pre-built scores from {p}')
            if p.suffix.lower() == '.csv':
                return pd.read_csv(p)
            return pd.read_json(p, lines=True)

        raise ValueError(
            "aa_lcr_pruned requires either:\n"
            "  (a) 'predictions_dir' + 'reviews_dir' pointing to the challenge repo's\n"
            "      Evals/Part 1/predictions and Evals/Part 1/reviews directories, OR\n"
            "  (b) 'scores_path' pointing to a pre-built flat CSV/JSONL.\n"
            "Set these in dataset-args when calling evalscope eval."
        )
