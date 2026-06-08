# flake8: noqa: E501
"""
mmmu_pruned_adapter.py
======================
evalscope adapter for the PRUNED MMMU variant.

Pruning is wired via PrunedAdapterMixin — the universal base that handles
the evalscope sample_filter hook and lazy index caching.  This class only
needs to implement _select_indices() with MMMU-specific logic.

Algorithm — Image-Type Encoder-Stress Pruner
--------------------------------------------
MMMU tests both vision-encoder capability and language reasoning. A model can
score well on MMMU by relying on language priors rather than actually perceiving
the image.

To stress the encoder specifically:
  1. Keep ALL high-stress samples (medical, chemical structure, geometric, etc.)
  2. Stratified sample of medium-stress samples (diagrams, charts, maps) by difficulty
  3. Keep Hard low-stress samples as calibration anchors

Key MMMU detail: integer 'index' is per-subset only (not globally unique).
The unique key is the string 'id' (e.g. 'validation_Accounting_23').
This adapter overrides _get_sample_key() to use the string id.

Parameters (extra_params)
--------------------------
predictions_dir       : dir of existing model prediction JSONL files (preferred)
medium_sample_frac    : float  [default 0.5]
keep_hard_low         : bool   [default True]
random_state          : int    [default 42]
"""
from typing import Any, List, Optional, Set

import pandas as pd

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope.benchmarks._pruned_mixin import PrunedAdapterMixin
from .mmmu_adapter import MMMUAdapter
from .pruner.image_type_pruner import ImageTypePruner

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE],
        description="""
## Overview

Pruned variant of MMMU for fast, cheap vision-encoder capability checks.

### Selection strategy: image-type encoder stress

Rather than running all 12,600 MMMU questions, we select the subset that
most directly exposes vision-encoder quality:

1. **HIGH-stress samples** (all kept) — questions requiring fine-grained visual
   perception: medical scans, chemical structures, geometric shapes, technical
   blueprints, microscopy. A model with a degraded encoder cannot answer these
   even if its language head is intact.

2. **MEDIUM-stress samples** (stratified sample by difficulty) — diagrams, charts,
   maps, sketches. Text context partially helps but image is still necessary.

3. **LOW-stress calibration anchors** (Hard difficulty only) — questions answerable
   partly from priors; retained as baseline calibration.

### Why this works for an unknown model

The subset is fixed from dataset metadata before the new model runs. It is
fully reproducible from `random_state` alone — no reference model scores needed.

### MMMU unique-key note

MMMU integer indices are per-subset and not globally unique.  This adapter
uses the string `id` (e.g. `validation_Accounting_23`) as the unique key.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `predictions_dir` | null | Dir of model prediction JSONL files for metadata loading |
| `medium_sample_frac` | 0.5 | Fraction of medium-stress samples per difficulty stratum |
| `keep_hard_low` | true | Retain Hard low-stress samples as calibration anchors |
| `random_state` | 42 | Reproducibility seed |
""",
        dataset_id='opencompass/MMMU',
        subset_list=[
            'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
            'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
            'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine',
            'Economics', 'Electronics', 'Energy_and_Power', 'Finance',
            'Geography', 'History', 'Literature', 'Manage',
            'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
            'Music', 'Pharmacy', 'Physics', 'Psychology',
            'Public_Health', 'Sociology',
        ],
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        extra_params={
            'predictions_dir': {
                'type': 'str | null',
                'description': (
                    'Directory of existing MMMU prediction JSONL files. '
                    'Used to load img_type, topic_difficulty, subfield metadata. '
                    'If null, metadata is read directly from the HuggingFace dataset.'
                ),
                'value': None,
            },
            'medium_sample_frac': {
                'type': 'float',
                'description': 'Fraction of medium-stress samples to keep per difficulty stratum.',
                'value': 0.5,
            },
            'keep_hard_low': {
                'type': 'bool',
                'description': 'Keep Hard-difficulty low-stress samples as calibration anchors.',
                'value': True,
            },
            'random_state': {
                'type': 'int',
                'description': 'Reproducibility seed.',
                'value': 42,
            },
        },
    )
)
class MMMUPrunedAdapter(PrunedAdapterMixin, MMMUAdapter):
    """
    MMMU adapter that evaluates only the encoder-stressing subset of questions.

    Inherits universal pruning plumbing from PrunedAdapterMixin and
    implements _select_indices() with MMMU image-type encoder-stress logic.

    Key override: _get_sample_key() returns the string 'id' (not int index)
    because MMMU integer indices are per-subset and not globally unique.
    """

    def __init__(self, benchmark_meta: BenchmarkMeta, task_config=None):
        super().__init__(benchmark_meta, task_config)
        self.__init_pruned_mixin__()

        self.predictions_dir: Optional[str] = self._get_extra_param('predictions_dir')
        self.medium_sample_frac: float      = float(self._get_extra_param('medium_sample_frac', 0.5))
        self.keep_hard_low: bool            = bool(self._get_extra_param('keep_hard_low', True))
        self.random_state: int              = int(self._get_extra_param('random_state', 42))

    # ── unique key override ────────────────────────────────────────────────────

    def _get_sample_key(self, sample: Sample) -> Optional[Any]:
        """Use the string 'id' as unique key — integer index is per-subset only."""
        return sample.metadata.get('id')

    # ── benchmark-specific selection logic ────────────────────────────────────

    def _select_indices(self) -> List[str]:
        """
        Returns a list of string sample IDs (e.g. 'validation_Accounting_23').

        Strategy:
          1. Load metadata (id, img_type, topic_difficulty, subfield).
          2. Run ImageTypePruner to select encoder-stressing samples.
        """
        meta_df = self._load_metadata()
        pruner = ImageTypePruner(
            medium_sample_frac=self.medium_sample_frac,
            keep_hard_low=self.keep_hard_low,
            random_state=self.random_state,
        )
        selected_ids = pruner.fit_transform(meta_df)
        logger.info(
            f'MMMUPrunedAdapter: selected {len(selected_ids)}/{len(meta_df)} samples '
            f'(frac_medium={self.medium_sample_frac}, keep_hard_low={self.keep_hard_low}).'
        )
        return selected_ids

    def _load_metadata(self) -> pd.DataFrame:
        """
        Build a DataFrame with columns: id, img_type, topic_difficulty, subfield.

        Preferred path: load from prediction JSONLs in predictions_dir.
        Fallback: stream the HuggingFace dataset (requires datasets library).
        """
        if self.predictions_dir:
            logger.info(f'MMMUPrunedAdapter: loading metadata from predictions at {self.predictions_dir}')
            return ImageTypePruner.load_metadata_from_predictions(self.predictions_dir)

        # Fallback: extract metadata from the live HuggingFace dataset.
        logger.info('MMMUPrunedAdapter: predictions_dir not set — loading metadata from HuggingFace dataset.')
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "The 'datasets' library is required when predictions_dir is not set. "
                "Install it with: pip install datasets"
            ) from e

        rows = []
        for subset in self.benchmark_meta.subset_list or []:
            try:
                ds = load_dataset(self.benchmark_meta.dataset_id, subset, split='validation', trust_remote_code=True)
                for rec in ds:
                    rows.append({
                        'id':               rec.get('id'),
                        'img_type':         rec.get('img_type'),
                        'topic_difficulty': rec.get('topic_difficulty'),
                        'subfield':         rec.get('subfield'),
                    })
            except Exception as exc:
                logger.warning(f'MMMUPrunedAdapter: skipping subset {subset}: {exc}')

        if not rows:
            raise ValueError('MMMUPrunedAdapter: no metadata rows could be loaded. Provide predictions_dir or check dataset access.')

        return pd.DataFrame(rows)
