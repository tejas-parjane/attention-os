"""Shared stratified split logic — one definition used by training and analysis.

Design: every row shares a single prediction date T0 (features strictly
before it, labels strictly after), so temporal separation holds by
construction. Users are then assigned to folds with seeded stratified
shuffle splits (``random_state=42``): disjoint users per fold, positive
rate preserved, exactly reproducible.

- :func:`outer_split` — 85% training pool / 15% held-out test.
- :func:`inner_split` — splits the training pool ~82% fit / ~18% validation
  (15 points of the full data). Model selection happens on this validation
  fold; the test fold is only ever reported, never used for decisions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42


def outer_split(
    X: pd.DataFrame, y: pd.Series, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices for (training pool, held-out test): 85 / 15."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    pool_idx, test_idx = next(splitter.split(X, y))
    return pool_idx, test_idx


def inner_split(
    X_pool: pd.DataFrame, y_pool: pd.Series, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (relative to the pool) for (fit, validation): ~82 / ~18."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=15 / 85, random_state=seed)
    fit_idx, val_idx = next(splitter.split(X_pool, y_pool))
    return fit_idx, val_idx
