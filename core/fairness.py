"""
fairness.py
───────────
Computes the three core group-fairness metrics used in FairLens:

    1. Demographic Parity Difference (DPD)
    2. Disparate Impact Ratio (DIR)
    3. Equal Opportunity Difference (EOD)

Each function is self-contained and works on plain numpy arrays so they
can also be called independently from the pipeline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class FairnessReport:
    """Structured container for all computed fairness metrics."""

    # Raw per-group stats
    positive_rate_group0: float = 0.0   # P(ŷ=1 | A=0)
    positive_rate_group1: float = 0.0   # P(ŷ=1 | A=1)
    tpr_group0: float = 0.0             # TPR for A=0
    tpr_group1: float = 0.0             # TPR for A=1

    # Fairness metrics
    demographic_parity_diff: float = 0.0
    disparate_impact_ratio: float = 0.0
    equal_opportunity_diff: float = 0.0

    # Verdict flags
    is_demographically_fair: bool = False   # |DPD| < 0.1
    is_disparate_impact_fair: bool = False  # DIR >= 0.8
    is_equal_opportunity_fair: bool = False # |EOD| < 0.1

    # Human-readable summary
    summary: Dict[str, str] = field(default_factory=dict)


# ─── Core metric functions ────────────────────────────────────────────────────

def demographic_parity_difference(y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Demographic Parity Difference (DPD).

        DPD = P(ŷ=1|A=1) − P(ŷ=1|A=0)

    Ideal value: 0  (equal approval rates across groups)
    Fair threshold: |DPD| < 0.1

    Args:
        y_pred: Predicted binary labels (0/1).
        A:      Sensitive attribute vector (0/1).

    Returns:
        Signed difference in positive prediction rates.
    """
    rate_0 = y_pred[A == 0].mean()
    rate_1 = y_pred[A == 1].mean()
    return float(rate_1 - rate_0)


def disparate_impact_ratio(y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Disparate Impact Ratio (DIR).

        DIR = P(ŷ=1|A=minority) / P(ŷ=1|A=majority)

    We treat A=0 as minority and A=1 as majority for convention.
    Ideal value: 1.0
    Legal threshold (80% rule): DIR >= 0.8 → considered fair.

    Args:
        y_pred: Predicted binary labels.
        A:      Sensitive attribute vector.

    Returns:
        Ratio of positive rates. Returns 0.0 if denominator is zero.
    """
    rate_0 = y_pred[A == 0].mean()   # minority (e.g. female)
    rate_1 = y_pred[A == 1].mean()   # majority (e.g. male)
    if rate_1 == 0:
        return 0.0
    return float(rate_0 / rate_1)


def equal_opportunity_difference(
    y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray
) -> float:
    """
    Equal Opportunity Difference (EOD).

        EOD = TPR(A=1) − TPR(A=0)
        TPR = TP / (TP + FN)   [True Positive Rate among ACTUAL positives]

    Ideal value: 0  (equal recall for qualified individuals)
    Fair threshold: |EOD| < 0.1

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        A:      Sensitive attribute vector.

    Returns:
        Signed difference in TPRs.
    """
    def tpr(mask):
        positives = y_true[mask] == 1
        if positives.sum() == 0:
            return 0.0
        return float(y_pred[mask][positives].mean())

    return tpr(A == 1) - tpr(A == 0)


# ─── Combined report ──────────────────────────────────────────────────────────

def compute_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A: np.ndarray,
    group_names: tuple = ("Group 0", "Group 1"),
) -> FairnessReport:
    """
    Compute all three fairness metrics and return a FairnessReport.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted binary labels.
        A:            Sensitive attribute vector (binary 0/1).
        group_names:  Human-readable names for groups 0 and 1.

    Returns:
        Populated FairnessReport dataclass.
    """
    dpd = demographic_parity_difference(y_pred, A)
    dir_ = disparate_impact_ratio(y_pred, A)
    eod = equal_opportunity_difference(y_true, y_pred, A)

    rate_0 = float(y_pred[A == 0].mean())
    rate_1 = float(y_pred[A == 1].mean())

    mask0, mask1 = A == 0, A == 1
    def _tpr(mask):
        pos = y_true[mask] == 1
        return float(y_pred[mask][pos].mean()) if pos.sum() else 0.0

    report = FairnessReport(
        positive_rate_group0=rate_0,
        positive_rate_group1=rate_1,
        tpr_group0=_tpr(mask0),
        tpr_group1=_tpr(mask1),
        demographic_parity_diff=dpd,
        disparate_impact_ratio=dir_,
        equal_opportunity_diff=eod,
        is_demographically_fair=abs(dpd) < 0.1,
        is_disparate_impact_fair=dir_ >= 0.8,
        is_equal_opportunity_fair=abs(eod) < 0.1,
    )

    # ── Human-readable summary ─────────────────────────────────────────────
    g0, g1 = group_names
    report.summary = {
        "demographic_parity": (
            f"{g0} approval rate: {rate_0:.2%} | "
            f"{g1} approval rate: {rate_1:.2%} | "
            f"DPD = {dpd:+.3f} → "
            f"{'✅ FAIR' if report.is_demographically_fair else '⚠️  BIASED'}"
        ),
        "disparate_impact": (
            f"DIR = {dir_:.3f} (≥0.8 is fair) → "
            f"{'✅ FAIR' if report.is_disparate_impact_fair else '⚠️  BIASED (below 80% rule)'}"
        ),
        "equal_opportunity": (
            f"TPR {g0}: {report.tpr_group0:.2%} | "
            f"TPR {g1}: {report.tpr_group1:.2%} | "
            f"EOD = {eod:+.3f} → "
            f"{'✅ FAIR' if report.is_equal_opportunity_fair else '⚠️  BIASED'}"
        ),
    }

    return report
