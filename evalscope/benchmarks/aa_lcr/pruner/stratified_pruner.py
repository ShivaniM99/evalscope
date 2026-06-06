"""
Stratified Discriminability Pruner for AA-LCR.
Selects the smallest subset that preserves model ranking,
defensible for unseen models via quadrant coverage guarantee.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class StratifiedDiscriminabilityPruner:
    """
    Two-stage pruner:
      Stage 1 — Keep all discriminative questions (acc_range > 0)
      Stage 2 — Stratified sample of non-discriminative questions
                by difficulty-complexity quadrant

    Parameters
    ----------
    quadrant_weights : dict
        Fraction of each quadrant to retain from non-discriminative pool.
        Defaults are tuned so hard/complex questions are prioritised.
    random_state : int
        Reproducibility seed.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "hard_complex": 1.00,
        "hard_simple":  0.25,
        "easy_complex": 1.00,
        "easy_simple":  0.05,
    }

    def __init__(
        self,
        quadrant_weights: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ):
        self.quadrant_weights = quadrant_weights or self.DEFAULT_WEIGHTS
        self.random_state = random_state

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def fit_transform(self, scores_df: pd.DataFrame) -> List[int]:
        """
        Parameters
        ----------
        scores_df : pd.DataFrame
            Must have columns: index, model, acc, reasoning_len, input_tokens
            One row per (index, model) — already deduplicated.

        Returns
        -------
        List[int] — selected question indices
        """
        per_q = self._build_features(scores_df)
        return self._select(per_q)

    # ──────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pq = (
            df.groupby("index")
            .agg(
                acc_range        =("acc", lambda x: x.max() - x.min()),
                acc_std          =("acc", "std"),
                mean_acc         =("acc", "mean"),
                median_reasoning =("reasoning_len", "median"),
                mean_input_tokens=("input_tokens", "mean"),
            )
            .reset_index()
        )

        # Discrimination flag
        pq["is_discriminative"] = pq["acc_range"] > 0

        # Difficulty score — outcome + disagreement
        acc_std_max = max(float(pq["acc_std"].max()), 1e-9)
        acc_std_norm = pq["acc_std"] / acc_std_max
        pq["difficulty_score"] = (1 - pq["mean_acc"]) * 0.6 + acc_std_norm * 0.4
        diff_max = max(float(pq["difficulty_score"].max()), 1e-9)
        pq["difficulty_score"] /= diff_max

        # Complexity — z-score of median reasoning length
        mu = pq["median_reasoning"].mean()
        sd = max(float(pq["median_reasoning"].std()), 1e-9)
        pq["reasoning_zscore"] = (pq["median_reasoning"] - mu) / sd

        # Quadrant classification
        hard = pq["difficulty_score"] > 0.5
        complex_ = pq["reasoning_zscore"] > 0.5
        pq["quadrant"] = np.select(
            [hard & complex_, hard & ~complex_,
             ~hard & complex_, ~hard & ~complex_],
            ["hard_complex", "hard_simple",
             "easy_complex", "easy_simple"],
            default="easy_simple",
        )
        return pq

    # ──────────────────────────────────────────────
    # Selection
    # ──────────────────────────────────────────────

    def _select(self, per_q: pd.DataFrame) -> List[int]:
        selected = set()

        # Stage 1: always keep all discriminative questions
        disc = per_q.loc[per_q["is_discriminative"], "index"].tolist()
        selected.update(disc)

        # Stage 2: stratified sample from non-discriminative pool
        non_disc = per_q[~per_q["is_discriminative"]]
        rng = np.random.default_rng(self.random_state)

        for quadrant, frac in self.quadrant_weights.items():
            group = non_disc[non_disc["quadrant"] == quadrant]
            if len(group) == 0:
                continue
            n = max(1, int(np.ceil(len(group) * frac)))
            n = min(n, len(group))
            sampled = group.sample(n=n, random_state=self.random_state)
            selected.update(sampled["index"].tolist())

        return sorted(selected)