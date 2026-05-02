"""
probe.py — Hallucination probe classifier (3-way ensemble).

The probe trains three diverse linear models on different *views* of the
flat feature vector emitted by ``aggregation.aggregate``, and averages their
class-1 probabilities at inference time.  Each member is its own
``StandardScaler -> PCA -> classifier`` pipeline, fit independently.

Members (sweep4 → see SOLUTION.md for details):

    M1  pools = (mean, last, lastK32)   PCA(48)   LogisticRegression(C=1.0)
    M2  pools = (lastK32, lastK64)      PCA(48)   LogisticRegression(C=1.0)
    M3  pools = (mean, last, lastK32)   PCA(64)   RidgeClassifier(alpha=100)

Why three members?
    * M1 is the best single LR by AUROC (sweep3); M2 the best single LR by
      accuracy on response-only views; M3 swaps the classifier (Ridge) and
      PCA dim for diversity.
    * Averaging their probabilities reduces seed-to-seed variance from
      ±3.10% to ±2.52% in 5-fold × 5-seed CV while *increasing* mean
      accuracy from 72.13% to 72.80%.

Public methods (signatures fixed by the contract): ``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``.

Threshold tuning (chosen empirically — see SOLUTION.md "Failure cases"):
    * Per-fold tuning (``fit_hyperparameters``): F1-best on the supplied
      validation split.  Empirically gave higher mean test accuracy across
      folds than accuracy-best (small 110-sample val sets make accuracy
      tuning unstable; F1's smoother landscape transfers better).
    * Final-probe seeding inside ``fit`` (no follow-up ``fit_hyperparameters``,
      e.g. solution.py's final-probe step): a 5-fold *out-of-fold* CV is
      run internally; the threshold is the **accuracy-best** value on the
      pooled OOF predictions.  This avoids the degenerate case where F1
      tuning on a single 138-sample holdout collapses to a near-zero
      threshold (predicting almost all-1s), which we observed in practice.

The class still extends ``torch.nn.Module`` (required by the contract) but
``forward`` is not used by the evaluator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import the layout helpers from aggregation so the probe knows how to slice X.
from aggregation import HIDDEN_DIM, POOL_ORDER, SELECTED_LAYERS, pool_indices

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_EXPECTED_AGG_DIM = len(SELECTED_LAYERS) * len(POOL_ORDER) * HIDDEN_DIM
_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Ensemble member spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MemberSpec:
    name: str
    pools: tuple[str, ...]
    pca_components: int
    clf_kind: str  # "logreg" | "ridge"
    clf_kwargs: dict


_MEMBERS: tuple[_MemberSpec, ...] = (
    _MemberSpec(
        name="M1_meanlastK32_pca48_lr",
        pools=("mean", "last", "lastK32"),
        pca_components=48,
        clf_kind="logreg",
        clf_kwargs=dict(C=1.0, max_iter=5000, class_weight="balanced",
                        solver="liblinear", random_state=_RANDOM_STATE),
    ),
    _MemberSpec(
        name="M2_lastK32_lastK64_pca48_lr",
        pools=("lastK32", "lastK64"),
        pca_components=48,
        clf_kind="logreg",
        clf_kwargs=dict(C=1.0, max_iter=5000, class_weight="balanced",
                        solver="liblinear", random_state=_RANDOM_STATE),
    ),
    _MemberSpec(
        name="M3_meanlastK32_pca64_ridge100",
        pools=("mean", "last", "lastK32"),
        pca_components=64,
        clf_kind="ridge",
        clf_kwargs=dict(alpha=100.0, class_weight="balanced",
                        random_state=_RANDOM_STATE),
    ),
)


def _make_clf(kind: str, **kwargs):
    if kind == "logreg":
        return LogisticRegression(**kwargs)
    if kind == "ridge":
        return RidgeClassifier(**kwargs)
    raise ValueError(f"unknown clf kind: {kind}")


# ---------------------------------------------------------------------------
# A single fitted ensemble member: scaler + (optional) PCA + classifier
# ---------------------------------------------------------------------------


class _FittedMember:
    __slots__ = ("spec", "scaler", "pca", "clf", "col_idx")

    def __init__(
        self,
        spec: _MemberSpec,
        scaler: StandardScaler,
        pca: PCA | None,
        clf,
        col_idx: np.ndarray,
    ) -> None:
        self.spec = spec
        self.scaler = scaler
        self.pca = pca
        self.clf = clf
        self.col_idx = col_idx

    @classmethod
    def fit(cls, spec: _MemberSpec, X_full: np.ndarray, y: np.ndarray) -> "_FittedMember":
        col_idx = _columns_for_pools(spec.pools)
        Xv = X_full[:, col_idx]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xv)
        pca = None
        if spec.pca_components is not None and spec.pca_components < min(Xs.shape):
            pca = PCA(n_components=spec.pca_components, random_state=_RANDOM_STATE)
            Xs = pca.fit_transform(Xs)
        clf = _make_clf(spec.clf_kind, **spec.clf_kwargs)
        clf.fit(Xs, y)
        return cls(spec, scaler, pca, clf, col_idx)

    def proba_pos(self, X_full: np.ndarray) -> np.ndarray:
        Xv = X_full[:, self.col_idx]
        Xs = self.scaler.transform(Xv)
        if self.pca is not None:
            Xs = self.pca.transform(Xs)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(Xs)[:, 1]
        # RidgeClassifier path → squash decision function with a sigmoid.
        s = self.clf.decision_function(Xs)
        return 1.0 / (1.0 + np.exp(-s))


def _columns_for_pools(pools: tuple[str, ...]) -> np.ndarray:
    cols: list[int] = []
    for p in pools:
        cols.extend(pool_indices(p))
    return np.asarray(cols, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public probe class
# ---------------------------------------------------------------------------


class HallucinationProbe(nn.Module):
    """3-way linear ensemble probe.

    See module docstring for the design rationale and member configuration.
    """

    def __init__(self) -> None:
        super().__init__()
        self._members: list[_FittedMember] = []
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_threshold_f1(probs: np.ndarray, y: np.ndarray) -> float:
        """Threshold maximising F1 on ``(probs, y)``."""
        cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_score = 0.5, -1.0
        for t in cands:
            yhat = (probs >= t).astype(int)
            score = f1_score(y, yhat, zero_division=0)
            if score > best_score:
                best_t, best_score = float(t), score
        return best_t

    @staticmethod
    def _best_threshold_acc(probs: np.ndarray, y: np.ndarray) -> float:
        """Threshold maximising accuracy on ``(probs, y)``.

        Ties are broken in favour of the threshold closest to 0.5 so the
        choice is well calibrated even when val accuracy plateaus over a
        wide range of thresholds.
        """
        cands = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_score = 0.5, -1.0
        for t in cands:
            yhat = (probs >= t).astype(int)
            score = accuracy_score(y, yhat)
            if score > best_score or (
                score == best_score and abs(t - 0.5) < abs(best_t - 0.5)
            ):
                best_t, best_score = float(t), score
        return best_t

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D feature matrix, got shape {X.shape}")
        # Allow the optional geometric tail to be appended without breaking
        # member views: we only ever index into the first _EXPECTED_AGG_DIM
        # columns (the pooled block) which is laid out exactly as agg.
        if X.shape[1] < _EXPECTED_AGG_DIM:
            raise ValueError(
                f"Feature dim {X.shape[1]} < expected pooled dim "
                f"{_EXPECTED_AGG_DIM}; check aggregation.py / probe.py "
                "are in sync."
            )
        return X

    def _ensemble_proba_pos(self, X_full: np.ndarray) -> np.ndarray:
        if not self._members:
            raise RuntimeError("HallucinationProbe is not fitted yet.")
        return np.mean(
            [m.proba_pos(X_full) for m in self._members], axis=0
        )

    def _seed_threshold_from_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Pick a robust threshold via 5-fold out-of-fold predictions.

        Called from ``fit`` when no validation set is supplied.  Inside this
        helper we run a *separate* stratified 5-fold CV: for each fold we
        train a fresh ensemble on the 4-fold train portion and predict on
        the held-out fold.  The OOF predictions cover all training samples
        without leakage, giving us ~``len(y)`` honest scores from which to
        pick an **accuracy-best** threshold.

        Using OOF on the full set (instead of a single 80/20 holdout + F1)
        avoids the degenerate threshold (~0.14, predicting almost all 1s)
        that F1-best produced on a small holdout in our case.

        The main ``self._members`` (already fit on all of ``X``) are *not*
        modified by this routine — only ``self._threshold`` is set.
        """
        if len(y) < 50:
            self._threshold = 0.5
            return
        try:
            n = len(y)
            oof_probs = np.zeros(n, dtype=np.float64)
            skf = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=_RANDOM_STATE
            )
            for idx_tr, idx_va in skf.split(np.arange(n), y):
                tmp_members = [
                    _FittedMember.fit(m.spec, X[idx_tr], y[idx_tr])
                    for m in self._members
                ]
                p_va = np.mean([m.proba_pos(X[idx_va]) for m in tmp_members], axis=0)
                oof_probs[idx_va] = p_va
            self._threshold = self._best_threshold_acc(oof_probs, y)
        except (ValueError, RuntimeError):
            self._threshold = 0.5

    # ------------------------------------------------------------------
    # Public contract — signatures fixed by the task spec.
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit every ensemble member on the full training set.

        Args:
            X: ``(n_samples, feature_dim)`` feature matrix.
            y: Integer label vector ``(n_samples,)``; 0=truthful, 1=halluc.

        Returns:
            ``self``.
        """
        X = self._validate_input(X)
        y = np.asarray(y).astype(int).ravel()

        self._members = [_FittedMember.fit(spec, X, y) for spec in _MEMBERS]
        self._seed_threshold_from_train(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on a validation split.

        Uses **F1-best** here because the val splits supplied by
        ``splitting.py`` are small (~110 samples) and accuracy-best
        thresholds are noisier on tiny val sets — F1's smoother landscape
        transferred better to test in our 5-fold sweep.  The complementary
        OOF / accuracy strategy is reserved for ``fit``-only callers (see
        ``_seed_threshold_from_train``).
        """
        if not self._members:
            raise RuntimeError("fit_hyperparameters called before fit().")
        X_val = self._validate_input(X_val)
        probs = self._ensemble_proba_pos(X_val)
        self._threshold = self._best_threshold_f1(probs, np.asarray(y_val).astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ``{0, 1}`` labels using the current threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n, 2)`` probability matrix; column 1 = P(hallucinated)."""
        X = self._validate_input(X)
        p1 = self._ensemble_proba_pos(X)
        return np.stack([1.0 - p1, p1], axis=1)

    # ------------------------------------------------------------------
    # nn.Module-compat shim (not used by evaluate.py)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe is an sklearn-based ensemble; the evaluator "
            "does not call forward()."
        )
