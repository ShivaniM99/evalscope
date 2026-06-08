# flake8: noqa: E501
"""
image_type_pruner.py
====================
Selects MMMU questions that specifically stress vision encoders rather than
language/reasoning capability.

Core insight
------------
A multimodal model has two separable components:
  - Vision encoder  → converts image pixels into tokens
  - Language head   → reasons over those tokens to produce an answer

A model can have a strong language head but a degraded encoder.  To detect
encoder failures we need questions where the *answer lives entirely in the image*
and cannot be inferred from text context, subject knowledge, or answer choices
alone.

Image types that stress encoders
---------------------------------
HIGH stress — answer requires fine-grained visual perception:
  • Medical scans (MRI, CT, X-rays, pathological images)
  • Chemical structures, DNA sequences
  • Technical blueprints, engineering schematics
  • Geometric shapes (spatial relationships must be preserved)
  • Scientific figures, microscopy

MEDIUM stress — image helps but text provides partial context:
  • Diagrams, trees/graphs
  • Maps
  • Sketches and drafts
  • Charts / figures with embedded labels

LOW stress — often answerable from text or prior knowledge:
  • Photographs, landscapes
  • Comics, cartoons
  • Logos, icons
  • Paintings

Selection strategy
------------------
1. Score each sample by encoder stress based on img_type.
2. Keep ALL high-stress samples.
3. Stratified sample of medium-stress samples by (subject × difficulty).
4. Thin out low-stress samples — keep only Hard ones for calibration.

This is model-agnostic: works on the full 12K HF dataset or the 660 reference
rows, because selection is based on content metadata, not model scores.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ── Encoder stress taxonomy ───────────────────────────────────────────────────

# Each entry is a substring that can appear in img_type strings.
# Order matters: first match wins.

HIGH_STRESS_PATTERNS: List[str] = [
    'MRI', 'CT scan', 'X-ray', 'Body Scan',
    'Pathological', 'Medical Image',
    'Chemical Structure',
    'DNA Sequence',
    'Technical Blueprint',
    'Geometric Shape',
    'Microscop',
    'Scientific Figure',
    'Electronics',          # circuit schematics
    'Spectro',              # spectroscopy charts
]

MEDIUM_STRESS_PATTERNS: List[str] = [
    'Diagram',
    'Tree', 'Graph',
    'Map',
    'Sketch', 'Draft',
    'Chart',
    'Figure',
    'Table',                # rendered as image
    'Plot',
]

# Anything not matching high/medium is treated as LOW stress.

# Subject-level encoder-stress boost: these subjects almost always require
# visual perception to answer correctly.
HIGH_STRESS_SUBJECTS: Set[str] = {
    'Basic_Medical_Science',
    'Clinical_Medicine',
    'Diagnostics_and_Laboratory_Medicine',
    'Biology',
    'Chemistry',
    'Physics',
    'Electronics',
    'Energy_and_Power',
    'Materials',
    'Architecture_and_Engineering',
    'Mechanical_Engineering',
    'Pharmacy',
    'Computer_Science',
}


def _stress_level(img_type_raw) -> str:
    """Return 'high', 'medium', or 'low' for an img_type value."""
    if img_type_raw is None:
        return 'low'

    # img_type may be stored as a list, a JSON string of a list, or a plain string
    if isinstance(img_type_raw, list):
        tags = img_type_raw
    elif isinstance(img_type_raw, str):
        s = img_type_raw.strip()
        if s.startswith('['):
            try:
                tags = ast.literal_eval(s)
            except Exception:
                tags = [s]
        else:
            tags = [s]
    else:
        tags = [str(img_type_raw)]

    combined = ' '.join(str(t) for t in tags).lower()

    for pat in HIGH_STRESS_PATTERNS:
        if pat.lower() in combined:
            return 'high'
    for pat in MEDIUM_STRESS_PATTERNS:
        if pat.lower() in combined:
            return 'medium'
    return 'low'


class ImageTypePruner:
    """
    Selects MMMU samples that stress the vision encoder.

    Parameters
    ----------
    medium_sample_frac : float
        Fraction of medium-stress samples to retain per (subject × difficulty)
        stratum.  Default 0.5.
    keep_hard_low : bool
        Whether to keep Hard difficulty low-stress samples as calibration
        anchors.  Default True.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        medium_sample_frac: float = 0.5,
        keep_hard_low: bool = True,
        random_state: int = 42,
    ):
        self.medium_sample_frac = medium_sample_frac
        self.keep_hard_low = keep_hard_low
        self.random_state = random_state

    # ── public API ────────────────────────────────────────────────────────────

    def fit_transform(self, meta_df: pd.DataFrame) -> List[str]:
        """
        Parameters
        ----------
        meta_df : pd.DataFrame
            One row per sample.  Required columns:
              id              — unique string ID (e.g. 'validation_Accounting_23')
              img_type        — raw img_type value (list or string)
              topic_difficulty — 'Easy' | 'Medium' | 'Hard'
              subfield        — subject subfield string

        Returns
        -------
        List[str] — selected sample IDs.
        """
        df = meta_df.copy()
        df['_stress'] = df['img_type'].apply(_stress_level)

        # Subject-level boost: upgrade to 'high' for encoder-heavy subjects
        if 'subfield' in df.columns:
            for subj in HIGH_STRESS_SUBJECTS:
                mask = df['subfield'].str.contains(subj, case=False, na=False)
                df.loc[mask & (df['_stress'] == 'medium'), '_stress'] = 'high'

        selected_ids: List[str] = []

        # Stage 1 — keep ALL high-stress samples
        high = df[df['_stress'] == 'high']
        selected_ids.extend(high['id'].tolist())

        # Stage 2 — stratified sample of medium-stress samples
        medium = df[df['_stress'] == 'medium']
        if not medium.empty:
            strat_col = 'topic_difficulty' if 'topic_difficulty' in medium.columns else None
            if strat_col:
                for _, grp in medium.groupby(strat_col):
                    n = max(1, int(len(grp) * self.medium_sample_frac))
                    sampled = grp.sample(n=min(n, len(grp)), random_state=self.random_state)
                    selected_ids.extend(sampled['id'].tolist())
            else:
                n = max(1, int(len(medium) * self.medium_sample_frac))
                selected_ids.extend(
                    medium.sample(n=n, random_state=self.random_state)['id'].tolist()
                )

        # Stage 3 — Hard low-stress samples as calibration anchors
        if self.keep_hard_low and 'topic_difficulty' in df.columns:
            hard_low = df[(df['_stress'] == 'low') & (df['topic_difficulty'] == 'Hard')]
            selected_ids.extend(hard_low['id'].tolist())

        # Deduplicate, preserve order
        seen: Set[str] = set()
        result = []
        for sid in selected_ids:
            if sid not in seen:
                seen.add(sid)
                result.append(sid)

        return result

    # ── convenience loader ────────────────────────────────────────────────────

    @staticmethod
    def load_metadata_from_predictions(predictions_dir: str) -> pd.DataFrame:
        """
        Load sample metadata from MMMU prediction JSONL files.

        Each JSONL row is expected to have:
          metadata.id, metadata.img_type, metadata.topic_difficulty,
          metadata.subfield

        Works for both the 660 reference rows and the full 12K dataset
        (if prediction files covering all samples are provided).
        """
        rows = []
        pred_path = Path(predictions_dir)
        # Support both flat dir (*.jsonl) and model subdirectory layout
        jsonl_files = list(pred_path.rglob('*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f'No JSONL files found under {predictions_dir}')

        seen_ids: Set[str] = set()
        for fpath in sorted(jsonl_files):
            with open(fpath, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    meta = rec.get('metadata', {})
                    sid = meta.get('id')
                    if sid and sid not in seen_ids:
                        seen_ids.add(sid)
                        rows.append({
                            'id':               sid,
                            'img_type':         meta.get('img_type'),
                            'topic_difficulty': meta.get('topic_difficulty'),
                            'subfield':         meta.get('subfield'),
                        })

        if not rows:
            raise ValueError(f'No valid MMMU prediction rows found in {predictions_dir}')

        return pd.DataFrame(rows)
