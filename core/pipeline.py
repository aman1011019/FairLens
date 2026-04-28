"""
pipeline.py
───────────
Main FairLens pipeline orchestrator.

Entry point:

    results = run_fairlens_pipeline(
        data           = df,           # pandas DataFrame
        sensitive_attr = "gender",     # protected attribute column
        target         = "income",     # label column
    )

Returns a fully populated FairLensResults dict ready for the backend API.
"""

import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional

from core.preprocessing import preprocess
from core.fairness import compute_fairness_report, FairnessReport
from core.counterfactual import run_counterfactual_analysis, CounterfactualReport
from core.explainability import compute_shap_explanation, ExplainabilityReport
from core.mitigation import (
    apply_reweighing,
    apply_threshold_adjustment,
    MitigationReport,
)


# ─── Pipeline result container ────────────────────────────────────────────────


class FairLensResults:
    """
    Structured result object returned by run_fairlens_pipeline().

    Attributes:
        meta               – run metadata (timing, dataset shape, …)
        baseline_metrics   – accuracy / classification report
        fairness_baseline  – FairnessReport before mitigation
        fairness_mitigated – FairnessReport after best mitigation
        counterfactual     – CounterfactualReport
        explainability     – ExplainabilityReport
        mitigation_rw      – MitigationReport (reweighing)
        mitigation_ta      – MitigationReport (threshold adjustment)
        verdict            – high-level bias verdict string
    """

    def __init__(self):
        self.meta: dict = {}
        self.baseline_metrics: dict = {}
        self.fairness_baseline: Optional[FairnessReport] = None
        self.fairness_mitigated: Optional[FairnessReport] = None
        self.counterfactual: Optional[CounterfactualReport] = None
        self.explainability: Optional[ExplainabilityReport] = None
        self.mitigation_rw: Optional[MitigationReport] = None
        self.mitigation_ta: Optional[MitigationReport] = None
        self.verdict: str = ""

    def to_dict(self) -> dict:
        """
        Serialise to a plain dict for JSON responses.
        SHAP values matrix is excluded (too large for JSON).
        """

        def _report_to_dict(r):
            if r is None:
                return None
            d = r.__dict__.copy()
            # Remove large numpy arrays that can't be JSON-serialised
            d.pop("shap_values", None)
            d.pop("sample_weights", None)
            d.pop("mitigated_model", None)
            # Serialise dataclass lists
            if "samples" in d:
                d["samples"] = [s.__dict__ for s in d["samples"]]
            if "global_importances" in d:
                d["global_importances"] = [f.__dict__ for f in d["global_importances"]]
            return d

        return {
            "meta": self.meta,
            "baseline_metrics": self.baseline_metrics,
            "fairness_baseline": _report_to_dict(self.fairness_baseline),
            "fairness_mitigated": _report_to_dict(self.fairness_mitigated),
            "counterfactual": _report_to_dict(self.counterfactual),
            "explainability": _report_to_dict(self.explainability),
            "mitigation_rw": _report_to_dict(self.mitigation_rw),
            "mitigation_ta": _report_to_dict(self.mitigation_ta),
            "verdict": self.verdict,
        }

    def print_summary(self):
        """Pretty-print a human-readable summary to stdout."""
        sep = "─" * 60
        print(f"\n{'═' * 60}")
        print("  F A I R L E N S   R E S U L T S")
        print(f"{'═' * 60}")
        print(f"\n📋 Meta")
        print(sep)
        for k, v in self.meta.items():
            print(f"  {k:30s}: {v}")

        print(f"\n📊 Baseline Accuracy")
        print(sep)
        for k, v in self.baseline_metrics.items():
            print(f"  {k:30s}: {v}")

        print(f"\n⚖️  Fairness — Baseline")
        print(sep)
        if self.fairness_baseline:
            for k, v in self.fairness_baseline.summary.items():
                print(f"  [{k}] {v}")

        print(f"\n⚖️  Fairness — After Mitigation")
        print(sep)
        if self.fairness_mitigated:
            for k, v in self.fairness_mitigated.summary.items():
                print(f"  [{k}] {v}")

        print(f"\n🔁 Counterfactual Fairness")
        print(sep)
        if self.counterfactual:
            print(f"  {self.counterfactual.summary}")

        print(f"\n🔍 Explainability (SHAP)")
        print(sep)
        if self.explainability:
            print(f"  {self.explainability.summary}")

        print(f"\n🔧 Mitigation — Reweighing")
        print(sep)
        if self.mitigation_rw:
            print(f"  {self.mitigation_rw.summary}")

        print(f"\n🔧 Mitigation — Threshold Adjustment")
        print(sep)
        if self.mitigation_ta:
            print(f"  {self.mitigation_ta.summary}")

        print(f"\n🏁 VERDICT")
        print(sep)
        print(f"  {self.verdict}")
        print(f"{'═' * 60}\n")


# ─── Main pipeline function ───────────────────────────────────────────────────


def run_fairlens_pipeline(
    data: pd.DataFrame,
    sensitive_attr: str = "gender",
    target: str = "income",
    test_size: float = 0.2,
    random_state: int = 42,
    run_shap: bool = True,
    run_counterfactual: bool = True,
    mitigation_strategy: str = "both",  # "reweighing" | "threshold" | "both"
    n_counterfactual_samples: int = 500,
) -> FairLensResults:
    """
    End-to-end FairLens pipeline.

    Pipeline steps:
        1. Preprocess  → (X_train, X_test, y_train, y_test, A_train, A_test)
        2. Train       → Logistic Regression baseline
        3. Fairness    → DPD, DIR, EOD on test set
        4. Counterfact → Flip-rate analysis
        5. SHAP        → Feature importance + bias signal
        6. Mitigate    → Reweighing + Threshold Adjustment
        7. Re-evaluate → Fairness metrics on mitigated model
        8. Verdict     → Aggregated bias verdict

    Args:
        data:                     Raw pandas DataFrame.
        sensitive_attr:           Protected attribute column name.
        target:                   Label column name.
        test_size:                Train/test split ratio.
        random_state:             Global random seed.
        run_shap:                 Whether to compute SHAP (slower).
        run_counterfactual:       Whether to run counterfactual analysis.
        mitigation_strategy:      Which mitigation(s) to apply.
        n_counterfactual_samples: Number of samples for counterfactual test.

    Returns:
        FairLensResults object.
    """
    results = FairLensResults()
    t_start = time.time()

    # ── STEP 1: Preprocess ────────────────────────────────────────────────────
    print("⏳ [1/8] Preprocessing...")
    prep = preprocess(
        data,
        sensitive_attr=sensitive_attr,
        target_col=target,
        test_size=test_size,
        random_state=random_state,
    )
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    A_train = prep["A_train"]
    A_test = prep["A_test"]
    feature_names = prep["feature_names"]
    group_names = tuple(prep["label_encoder"].classes_)  # e.g. ("Female", "Male")

    results.meta = {
        "sensitive_attribute": sensitive_attr,
        "target": target,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(feature_names),
        "group_0": group_names[0],
        "group_1": group_names[1],
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    # ── STEP 2: Train baseline ────────────────────────────────────────────────
    print("⏳ [2/8] Training baseline Logistic Regression...")
    baseline_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        C=1.0,
    )
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    results.baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_baseline)),
        "classification_report": classification_report(y_test, y_pred_baseline),
    }

    # ── STEP 3: Fairness metrics — Baseline ───────────────────────────────────
    print("⏳ [3/8] Computing fairness metrics (baseline)...")
    results.fairness_baseline = compute_fairness_report(
        y_test, y_pred_baseline, A_test, group_names=group_names
    )

    # ── STEP 4: Counterfactual analysis ───────────────────────────────────────
    if run_counterfactual:
        print("⏳ [4/8] Running counterfactual analysis...")
        results.counterfactual = run_counterfactual_analysis(
            model=baseline_model,
            X_test=X_test,
            A_test=A_test,
            feature_names=feature_names,
            sensitive_attr=sensitive_attr,
            n_samples=n_counterfactual_samples,
            random_state=random_state,
        )
    else:
        print("⏩ [4/8] Counterfactual skipped.")

    # ── STEP 5: SHAP explainability ───────────────────────────────────────────
    if run_shap:
        print("⏳ [5/8] Computing SHAP explanations...")
        results.explainability = compute_shap_explanation(
            model=baseline_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            sensitive_attr=sensitive_attr,
        )
    else:
        print("⏩ [5/8] SHAP skipped.")

    # ── STEP 6: Mitigation ────────────────────────────────────────────────────
    best_mitigated_model = baseline_model  # default fallback

    if mitigation_strategy in ("reweighing", "both"):
        print("⏳ [6/8] Applying reweighing mitigation...")
        results.mitigation_rw = apply_reweighing(
            base_model=LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=random_state
            ),
            X_train=X_train,
            y_train=y_train,
            A_train=A_train,
        )
        best_mitigated_model = results.mitigation_rw.mitigated_model

    if mitigation_strategy in ("threshold", "both"):
        print("⏳ [6/8] Applying threshold adjustment...")
        results.mitigation_ta = apply_threshold_adjustment(
            base_model=baseline_model,
            X_val=X_test,
            y_val=y_test,
            A_val=A_test,
            criterion="equal_opportunity",
        )
        # Prefer reweighing model for re-evaluation (simpler interface)

    # ── STEP 7: Re-evaluate fairness after mitigation ─────────────────────────
    print("⏳ [7/8] Re-evaluating fairness after mitigation...")
    if results.mitigation_rw is not None:
        y_pred_mitigated = results.mitigation_rw.mitigated_model.predict(X_test)
    elif results.mitigation_ta is not None:
        y_pred_mitigated = results.mitigation_ta.mitigated_model.predict(X_test, A_test)
    else:
        y_pred_mitigated = y_pred_baseline

    results.fairness_mitigated = compute_fairness_report(
        y_test, y_pred_mitigated, A_test, group_names=group_names
    )

    # ── STEP 8: Verdict ───────────────────────────────────────────────────────
    print("⏳ [8/8] Generating verdict...")
    results.verdict = _build_verdict(results)

    t_end = time.time()
    results.meta["runtime_seconds"] = round(t_end - t_start, 2)
    print(f"✅ Pipeline complete in {results.meta['runtime_seconds']}s.")

    return results


# ─── Verdict builder ──────────────────────────────────────────────────────────


def _build_verdict(r: FairLensResults) -> str:
    """Aggregate all signals into a plain-language bias verdict."""
    signals = []
    improvements = []

    fb = r.fairness_baseline
    if fb:
        if not fb.is_disparate_impact_fair:
            signals.append(
                f"Disparate Impact ({fb.disparate_impact_ratio:.2f}) is below the 80% rule."
            )
        if not fb.is_demographically_fair:
            signals.append(
                f"Demographic Parity Difference ({fb.demographic_parity_diff:+.3f}) exceeds threshold."
            )
        if not fb.is_equal_opportunity_fair:
            signals.append(
                f"Equal Opportunity Difference ({fb.equal_opportunity_diff:+.3f}) exceeds threshold."
            )

    if r.counterfactual and not r.counterfactual.is_counterfactually_fair:
        signals.append(
            f"Counterfactual Flip Rate ({r.counterfactual.flip_rate:.2%}) — "
            f"model changes decisions when identity is flipped."
        )

    if r.explainability and r.explainability.bias_signal_detected:
        signals.append(
            f"SHAP: Sensitive attribute '{r.meta.get('sensitive_attribute')}' "
            f"has notable influence on predictions."
        )

    fm = r.fairness_mitigated
    if fm and fb:
        old_dir = fb.disparate_impact_ratio
        new_dir = fm.disparate_impact_ratio
        if new_dir > old_dir:
            improvements.append(
                f"Disparate Impact improved {old_dir:.3f} → {new_dir:.3f} after mitigation."
            )

    if not signals:
        verdict = "✅ No significant bias detected. The model appears fair w.r.t. the sensitive attribute."
    else:
        verdict = f"⚠️  BIAS DETECTED ({len(signals)} signal(s)): " + " | ".join(signals)
        if improvements:
            verdict += " | Improvements after mitigation: " + " | ".join(improvements)

    return verdict
