"""
splitting.py — Train / validation / test split utilities.

Strategy
--------
Stratified K-fold cross-validation with a held-out validation slice carved out
of each fold's training portion (used only for threshold tuning inside
``probe.fit_hyperparameters``).  The contract returned to ``solution.py`` is a
list of ``(idx_train, idx_val, idx_test)`` tuples — one per fold.

Why K-fold and not a single split?
    With only 689 samples a single 70/15/15 split has high variance in the
    test estimate.  Averaging metrics across 5 folds makes the reported
    accuracy / F1 / AUROC meaningfully more stable, which is essential when
    we iterate on aggregation and probe design.

Why a separate val slice inside each fold?
    ``evaluate.run_evaluation`` calls ``probe.fit_hyperparameters(X_val, y_val)``
    after ``probe.fit``.  The val slice is *only* used for decision-threshold
    tuning, not for selecting features or fitting the scaler — so it does not
    leak signal back into training.

Reproducibility:
    A single base ``random_state`` controls both the outer fold partition and
    the inner train/val carve-out for every fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

N_SPLITS = 5
INNER_VAL_FRACTION = 0.20  # 20% of each fold's train portion -> val


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a list of ``(idx_train, idx_val, idx_test)`` per fold.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (unused; kept for API contract).
        test_size:    Unused — kept for backward compatibility with the API.
        val_size:     Unused — fixed at ``INNER_VAL_FRACTION`` of each fold.
        random_state: Base seed; each fold's inner split is offset from this.

    Returns:
        List of length ``N_SPLITS``; one tuple of integer index arrays per
        fold.  Across folds, every sample appears in exactly one ``idx_test``.
    """
    del df, test_size, val_size

    y = np.asarray(y).astype(int).ravel()
    idx = np.arange(len(y))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_id, (idx_train_full, idx_test) in enumerate(skf.split(idx, y)):
        idx_train, idx_val = train_test_split(
            idx_train_full,
            test_size=INNER_VAL_FRACTION,
            random_state=random_state + fold_id,
            stratify=y[idx_train_full],
        )
        splits.append((np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)))

    return splits
