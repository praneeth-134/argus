import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  -> no significant shift
    0.1 - 0.2  -> moderate shift
    > 0.2      -> significant shift, model likely needs retraining
    """
    # define bucket edges based on baseline distribution
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(baseline, breakpoints)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bucket_edges)
    current_counts, _ = np.histogram(current, bins=bucket_edges)

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    # avoid division by zero / log(0)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    return float(np.sum(psi_values))


def compute_ks_test(baseline: np.ndarray, current: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test for distribution shift."""
    statistic, p_value = ks_2samp(baseline, current)
    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_shift": bool(p_value < 0.05)
    }


def compute_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Compute drift for every feature.
    Returns: {feature_name: {psi, ks_statistic, p_value, drifted}}
    """
    results = {}
    for feature in feature_names:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue

        baseline_values = baseline_df[feature].dropna().values
        current_values = current_df[feature].dropna().values

        if len(baseline_values) < 5 or len(current_values) < 5:
            continue  # not enough data to compute meaningful drift

        psi = compute_psi(baseline_values, current_values)
        ks_result = compute_ks_test(baseline_values, current_values)

        results[feature] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks_result["ks_statistic"], 4),
            "p_value": round(ks_result["p_value"], 4),
            "drifted": psi > 0.2 or ks_result["significant_shift"]
        }

    return results