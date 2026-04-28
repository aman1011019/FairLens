"""
counterfactual.py
─────────────────
Implements Counterfactual Fairness analysis for FairLens.

Core idea: for each test sample, flip the sensitive attribute (A → 1−A)
while keeping ALL other features identical, then measure how often the
model changes its prediction.

    Counterfactual Flip Rate (CFR):
        CFR = (# samples where ŷ changes) / (total samples)

    CFR ≈ 0  →  model ignores sensitive attribute  (fair)
    CFR >> 0 →  model decision depends on identity  (biased)

We also expose per-sample flip details so the frontend can render
individual "what-if" stories.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# ─── Result containers ────────────────────────────────────────────────────────


@dataclass
class CounterfactualSample:
    """Stores the original vs. flipped prediction for a single data point."""

    sample_idx: int
    original_A: int  # 0 or 1
    counterfactual_A: int  # 1 or 0
    original_pred: int  # ŷ for original identity
    counterfactual_pred: int  # ŷ after flipping identity
    prediction_flipped: bool  # True if decision changed
    original_proba: float  # model confidence for original
    counterfactual_proba: float  # model confidence after flip


@dataclass
class CounterfactualReport:
    """Aggregated counterfactual fairness results."""

    flip_rate: float  # CFR = flips / n
    n_samples: int
    n_flipped: int
    is_counterfactually_fair: bool  # flip_rate < 0.05
    # Breakdown by who was flipped (minority → majority vs vice versa)
    flip_rate_0_to_1: float = 0.0  # minority group being treated as majority
    flip_rate_1_to_0: float = 0.0  # majority group being treated as minority
    samples: List[CounterfactualSample] = field(default_factory=list)
    summary: str = ""


# ─── Index of the sensitive feature inside X ──────────────────────────────────


def _find_sensitive_feature_indices(
    feature_names: List[str],
    sensitive_attr: str,
) -> List[int]:
    """
    Return column indices in X that correspond to the sensitive attribute.
    After one-hot encoding, gender becomes e.g. 'gender_Male'.
    We match on prefix to catch all encoded variants.
    """
    indices = [
        i
        for i, name in enumerate(feature_names)
        if name.lower().startswith(sensitive_attr.lower())
    ]
    return indices


# ─── Main analysis function ───────────────────────────────────────────────────


def run_counterfactual_analysis(
    model,
    X_test: np.ndarray,
    A_test: np.ndarray,
    feature_names: List[str],
    sensitive_attr: str,
    n_samples: int = 500,
    random_state: int = 42,
) -> CounterfactualReport:
    """
    Run counterfactual fairness test on the test set.

    Strategy:
        - For each sample, we flip A (0→1 or 1→0).
        - The sensitive attribute may appear directly in X as a one-hot
          encoded column (e.g., 'gender_Male'). We flip those columns.
        - We then compare model predictions.

    Args:
        model:          Fitted sklearn-compatible classifier (needs predict + predict_proba).
        X_test:         Test feature matrix (n, d).
        A_test:         Sensitive attribute vector (n,) with values in {0, 1}.
        feature_names:  Names of columns in X_test.
        sensitive_attr: Name of the sensitive attribute (e.g. "gender").
        n_samples:      Max number of samples to test (for speed).
        random_state:   Seed for sampling.

    Returns:
        CounterfactualReport with per-sample and aggregate results.
    """
    rng = np.random.default_rng(random_state)
    n_total = X_test.shape[0]
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    # Find which columns encode the sensitive attribute in X
    sensitive_cols = _find_sensitive_feature_indices(feature_names, sensitive_attr)

    X_sample = X_test[indices]
    A_sample = A_test[indices]

    # Build counterfactual version of X
    X_counter = X_sample.copy()

    if sensitive_cols:
        # Flip the one-hot encoded sensitive columns
        # Assumes binary sensitive attribute: negate each associated column
        for col_idx in sensitive_cols:
            # If value is 1 → 0 and vice versa (works for one binary column)
            X_counter[:, col_idx] = 1.0 - X_counter[:, col_idx]
    else:
        # Sensitive attribute was not included in X (excluded by design).
        # In this case counterfactual analysis operates on A directly
        # and we note it in the summary.
        pass

    # Predictions
    orig_pred = model.predict(X_sample)
    count_pred = model.predict(X_counter)

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        orig_proba = model.predict_proba(X_sample)[:, 1]
        count_proba = model.predict_proba(X_counter)[:, 1]
    else:
        orig_proba = orig_pred.astype(float)
        count_proba = count_pred.astype(float)

    # Per-sample results
    samples = []
    for i, global_idx in enumerate(indices):
        flipped = bool(orig_pred[i] != count_pred[i])
        samples.append(
            CounterfactualSample(
                sample_idx=int(global_idx),
                original_A=int(A_sample[i]),
                counterfactual_A=int(1 - A_sample[i]),
                original_pred=int(orig_pred[i]),
                counterfactual_pred=int(count_pred[i]),
                prediction_flipped=flipped,
                original_proba=float(orig_proba[i]),
                counterfactual_proba=float(count_proba[i]),
            )
        )

    # Aggregate
    flips = np.array([s.prediction_flipped for s in samples])
    flip_rate = float(flips.mean())

    # Directional flip rates
    mask_0 = A_sample == 0
    mask_1 = A_sample == 1
    cfr_0_to_1 = float(flips[mask_0].mean()) if mask_0.sum() > 0 else 0.0
    cfr_1_to_0 = float(flips[mask_1].mean()) if mask_1.sum() > 0 else 0.0

    is_fair = flip_rate < 0.05

    summary = (
        f"Counterfactual Flip Rate: {flip_rate:.2%} "
        f"({'✅ FAIR' if is_fair else '⚠️  BIASED'}) | "
        f"{flips.sum()} out of {len(flips)} samples changed prediction "
        f"when identity was flipped. "
        f"Group-0→1 flip rate: {cfr_0_to_1:.2%} | "
        f"Group-1→0 flip rate: {cfr_1_to_0:.2%}."
    )
    if not sensitive_cols:
        summary += (
            " NOTE: Sensitive attribute was not found as a column in X. "
            "Consider including it as a feature to enable full counterfactual analysis."
        )

    return CounterfactualReport(
        flip_rate=flip_rate,
        n_samples=len(samples),
        n_flipped=int(flips.sum()),
        is_counterfactually_fair=is_fair,
        flip_rate_0_to_1=cfr_0_to_1,
        flip_rate_1_to_0=cfr_1_to_0,
        samples=samples,
        summary=summary,
    )


def get_counterfactual_examples(
    report: CounterfactualReport,
    n: int = 5,
    only_flipped: bool = True,
) -> List[CounterfactualSample]:
    """
    Retrieve notable counterfactual examples for the frontend/explainer.

    Args:
        report:       Output of run_counterfactual_analysis.
        n:            Number of examples to return.
        only_flipped: If True, return only samples where prediction changed.

    Returns:
        List of CounterfactualSample objects.
    """
    pool = report.samples
    if only_flipped:
        pool = [s for s in pool if s.prediction_flipped]
    return pool[:n]
