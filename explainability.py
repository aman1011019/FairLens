"""
explainability.py
─────────────────
SHAP-based model explainability for FairLens.

Answers two questions:
    1. GLOBAL: Which features drive the model's decisions overall?
    2. BIAS SIGNAL: Is the sensitive attribute (e.g. gender) among the
                   top influential features? If yes → direct bias signal.

Uses shap.LinearExplainer for Logistic Regression (exact, fast).
Falls back to shap.KernelExplainer for any other sklearn model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─── Result containers ────────────────────────────────────────────────────────


@dataclass
class FeatureImportance:
    """Mean |SHAP| value for a single feature."""

    feature_name: str
    mean_abs_shap: float
    rank: int


@dataclass
class ExplainabilityReport:
    """SHAP-based explainability results."""

    global_importances: List[FeatureImportance] = field(default_factory=list)
    # SHAP values matrix (n_samples × n_features) for the positive class
    shap_values: Optional[np.ndarray] = None
    # Top-k feature names
    top_features: List[str] = field(default_factory=list)
    # Bias-specific: does sensitive attribute appear in top-k?
    sensitive_feature_rank: Optional[int] = None
    sensitive_mean_abs_shap: Optional[float] = None
    sensitive_is_top_k: bool = False  # True if appears in top-10
    bias_signal_detected: bool = False  # True if sensitive_mean_abs_shap > threshold
    summary: str = ""
    shap_available: bool = True


# ─── Main explainer ───────────────────────────────────────────────────────────


def compute_shap_explanation(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    sensitive_attr: str,
    n_background: int = 100,
    n_explain: int = 300,
    bias_shap_threshold: float = 0.05,
) -> ExplainabilityReport:
    """
    Compute SHAP values and build an ExplainabilityReport.

    Args:
        model:               Fitted sklearn classifier.
        X_train:             Training data (used as SHAP background).
        X_test:              Test data to explain.
        feature_names:       Feature column names (len == X.shape[1]).
        sensitive_attr:      Name of the protected attribute (e.g. "gender").
        n_background:        # background samples for KernelExplainer.
        n_explain:           # test samples to compute SHAP for.
        bias_shap_threshold: Mean |SHAP| above this → bias signal flagged.

    Returns:
        ExplainabilityReport
    """
    if not SHAP_AVAILABLE:
        return ExplainabilityReport(
            shap_available=False, summary="⚠️  SHAP not installed. Run: pip install shap"
        )

    from sklearn.linear_model import LogisticRegression

    # ── Choose explainer ───────────────────────────────────────────────────
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(
            model, X_train, feature_perturbation="interventional"
        )
    else:
        background = shap.sample(X_train, min(n_background, X_train.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    X_explain = X_test[: min(n_explain, X_test.shape[0])]

    # ── Compute SHAP values ────────────────────────────────────────────────
    raw = explainer.shap_values(X_explain)

    # LinearExplainer returns array of shape (n, d) for binary classification
    # KernelExplainer returns list [shap_class0, shap_class1]
    if isinstance(raw, list):
        shap_vals = raw[1]  # positive class
    else:
        shap_vals = raw

    # ── Global importances (mean |SHAP|) ───────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)  # (d,)
    ranked_idx = np.argsort(mean_abs)[::-1]  # descending

    importances = []
    for rank, idx in enumerate(ranked_idx, start=1):
        importances.append(
            FeatureImportance(
                feature_name=feature_names[idx],
                mean_abs_shap=float(mean_abs[idx]),
                rank=rank,
            )
        )

    top_features = [imp.feature_name for imp in importances[:10]]

    # ── Sensitive attribute contribution ──────────────────────────────────
    # The sensitive attribute may appear as one-hot columns in feature_names
    sensitive_indices = [
        i
        for i, name in enumerate(feature_names)
        if name.lower().startswith(sensitive_attr.lower())
    ]

    if sensitive_indices:
        # Sum SHAP across all encoded columns of the sensitive attr
        sensitive_shap = np.abs(shap_vals[:, sensitive_indices]).sum(axis=1)
        s_mean = float(sensitive_shap.mean())

        # Rank of the sensitive attribute group among all features
        # (treat as a single aggregated feature)
        other_means = np.delete(mean_abs, sensitive_indices)
        s_rank = int((other_means > s_mean).sum()) + 1  # 1-indexed

        s_top_k = s_rank <= 10
        s_bias = s_mean > bias_shap_threshold
    else:
        s_mean = None
        s_rank = None
        s_top_k = False
        s_bias = False

    # ── Summary string ─────────────────────────────────────────────────────
    top3 = ", ".join(top_features[:3])
    summary_parts = [
        f"Top 3 most influential features: {top3}.",
    ]
    if s_mean is not None:
        summary_parts.append(
            f"Sensitive attribute '{sensitive_attr}' → "
            f"mean |SHAP| = {s_mean:.4f} (rank #{s_rank}). "
            f"{'⚠️  BIAS SIGNAL DETECTED' if s_bias else '✅ Low influence on decisions'}."
        )
    else:
        summary_parts.append(
            f"Sensitive attribute '{sensitive_attr}' was not found among feature columns "
            f"(may have been excluded from X). No direct SHAP bias signal available."
        )

    return ExplainabilityReport(
        global_importances=importances,
        shap_values=shap_vals,
        top_features=top_features,
        sensitive_feature_rank=s_rank,
        sensitive_mean_abs_shap=s_mean,
        sensitive_is_top_k=s_top_k,
        bias_signal_detected=s_bias,
        summary=" ".join(summary_parts),
        shap_available=True,
    )


def get_local_explanation(
    shap_report: ExplainabilityReport,
    sample_idx: int,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the top-k most influential features for a single sample.

    Args:
        shap_report:   Output of compute_shap_explanation.
        sample_idx:    Index into the explained samples (0 to n_explain-1).
        feature_names: Feature names list.
        top_k:         Number of features to return.

    Returns:
        List of (feature_name, shap_value) sorted by |shap_value| descending.
    """
    if shap_report.shap_values is None:
        return []

    vals = shap_report.shap_values[sample_idx]
    ranked = np.argsort(np.abs(vals))[::-1][:top_k]
    return [(feature_names[i], float(vals[i])) for i in ranked]
