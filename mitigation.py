"""
mitigation.py
─────────────
Implements two bias mitigation strategies for FairLens:

    1. PRE-PROCESSING  – Reweighing
       Adjust sample weights before training so the model sees a
       "balanced" distribution across (sensitive attribute, label) groups.

    2. POST-PROCESSING – Threshold Adjustment (per-group optimal threshold)
       After training on the original model, find decision thresholds
       for each sensitive group that equalise a chosen fairness criterion
       (Equal Opportunity or Demographic Parity).

Both strategies return a re-trained or re-calibrated model plus a
MitigationReport so the pipeline can compare before vs. after metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.base import clone


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class MitigationReport:
    """Stores the mitigated model and its training metadata."""

    strategy: str  # "reweighing" | "threshold"
    mitigated_model: object  # fitted model (or wrapper)
    sample_weights: Optional[np.ndarray]  # reweighing: weights array; else None
    thresholds: Optional[Dict[int, float]]  # threshold: {group → threshold}; else None
    # Accuracy on training set with mitigation
    train_accuracy: float = 0.0
    summary: str = ""


# ─── Strategy 1: Reweighing ───────────────────────────────────────────────────


def compute_reweighing_weights(
    y_train: np.ndarray,
    A_train: np.ndarray,
) -> np.ndarray:
    """
    Compute sample weights that balance the (A, y) joint distribution.

    Formula:
        w(A=a, y=c) = P(A=a) * P(y=c) / P(A=a, y=c)
                    = Expected / Observed

    Groups that are under-represented in positive predictions get
    higher weights, nudging the model toward demographic parity.

    Args:
        y_train: Training labels (0/1).
        A_train: Training sensitive attributes (0/1).

    Returns:
        Sample weight array of shape (n,).
    """
    n = len(y_train)
    weights = np.ones(n, dtype=float)

    # Marginal probabilities
    p_a = {v: (A_train == v).mean() for v in [0, 1]}
    p_y = {v: (y_train == v).mean() for v in [0, 1]}

    for a_val in [0, 1]:
        for y_val in [0, 1]:
            mask = (A_train == a_val) & (y_train == y_val)
            p_joint = mask.mean()
            if p_joint > 0:
                w = (p_a[a_val] * p_y[y_val]) / p_joint
                weights[mask] = w

    # Normalise so weights sum to n (keeps effective sample size)
    weights = weights / weights.mean()
    return weights


def apply_reweighing(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    A_train: np.ndarray,
) -> MitigationReport:
    """
    Train a clone of base_model using reweighing sample weights.

    Args:
        base_model: Unfitted (or fitted) sklearn-compatible model.
        X_train:    Training features.
        y_train:    Training labels.
        A_train:    Training sensitive attributes.

    Returns:
        MitigationReport with the reweighed model.
    """
    weights = compute_reweighing_weights(y_train, A_train)

    mitigated = clone(base_model)
    mitigated.fit(X_train, y_train, sample_weight=weights)

    train_acc = (mitigated.predict(X_train) == y_train).mean()

    summary = (
        f"Reweighing applied. "
        f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]. "
        f"Train accuracy after mitigation: {train_acc:.3f}."
    )

    return MitigationReport(
        strategy="reweighing",
        mitigated_model=mitigated,
        sample_weights=weights,
        thresholds=None,
        train_accuracy=float(train_acc),
        summary=summary,
    )


# ─── Strategy 2: Per-Group Threshold Adjustment ───────────────────────────────


def find_equal_opportunity_thresholds(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    A_val: np.ndarray,
    criterion: str = "equal_opportunity",
) -> Dict[int, float]:
    """
    Find per-group decision thresholds that equalise TPR across groups.

    Method:
        For each group, sweep thresholds [0.01, 0.99] and find the one
        whose TPR is closest to the global average TPR.

    Args:
        model:     Fitted model with predict_proba.
        X_val:     Validation features.
        y_val:     Validation labels.
        A_val:     Validation sensitive attributes.
        criterion: "equal_opportunity" (match TPR) or
                   "demographic_parity" (match positive rate).

    Returns:
        Dict mapping group value → optimal threshold.
    """
    probas = model.predict_proba(X_val)[:, 1]
    thresholds_grid = np.linspace(0.01, 0.99, 99)

    def metric_at_threshold(mask, t):
        preds = (probas[mask] >= t).astype(int)
        if criterion == "equal_opportunity":
            positives = y_val[mask] == 1
            return float(preds[positives].mean()) if positives.sum() else 0.0
        else:  # demographic_parity
            return float(preds.mean())

    # Global target metric
    global_target = metric_at_threshold(np.ones(len(A_val), dtype=bool), 0.5)

    optimal_thresholds = {}
    for group in [0, 1]:
        mask = A_val == group
        best_t = 0.5
        best_diff = float("inf")
        for t in thresholds_grid:
            m = metric_at_threshold(mask, t)
            diff = abs(m - global_target)
            if diff < best_diff:
                best_diff = diff
                best_t = t
        optimal_thresholds[group] = float(best_t)

    return optimal_thresholds


class ThresholdAdjustedClassifier:
    """
    Thin wrapper that applies per-group thresholds at prediction time.

    Compatible with sklearn's predict / predict_proba interface.
    """

    def __init__(self, base_model, thresholds: Dict[int, float]):
        """
        Args:
            base_model: Fitted probability-calibrated sklearn model.
            thresholds: {group_value → threshold}.
        """
        self.base_model = base_model
        self.thresholds = thresholds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Predict using per-group thresholds.

        Args:
            X: Feature matrix.
            A: Sensitive attribute vector (same length as X).

        Returns:
            Binary predictions.
        """
        probas = self.base_model.predict_proba(X)[:, 1]
        preds = np.zeros(len(X), dtype=int)
        for group, thresh in self.thresholds.items():
            mask = A == group
            preds[mask] = (probas[mask] >= thresh).astype(int)
        return preds

    # Convenience: allow predict(X) without A for pipeline compatibility
    def predict_no_group(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X)


def apply_threshold_adjustment(
    base_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    A_val: np.ndarray,
    criterion: str = "equal_opportunity",
) -> MitigationReport:
    """
    Compute and wrap per-group thresholds into a ThresholdAdjustedClassifier.

    Args:
        base_model: Already-fitted model.
        X_val:      Validation/test features for threshold search.
        y_val:      Validation labels.
        A_val:      Validation sensitive attributes.
        criterion:  "equal_opportunity" | "demographic_parity".

    Returns:
        MitigationReport with the wrapped model and threshold map.
    """
    thresholds = find_equal_opportunity_thresholds(
        base_model, X_val, y_val, A_val, criterion
    )

    wrapped = ThresholdAdjustedClassifier(base_model, thresholds)
    preds = wrapped.predict(X_val, A_val)
    train_acc = float((preds == y_val).mean())

    summary = (
        f"Threshold adjustment ({criterion}) applied. "
        f"Group-0 threshold: {thresholds[0]:.3f} | "
        f"Group-1 threshold: {thresholds[1]:.3f}. "
        f"Validation accuracy: {train_acc:.3f}."
    )

    return MitigationReport(
        strategy=f"threshold_adjustment_{criterion}",
        mitigated_model=wrapped,
        sample_weights=None,
        thresholds=thresholds,
        train_accuracy=train_acc,
        summary=summary,
    )
