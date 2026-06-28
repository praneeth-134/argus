import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def explain_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Drift attribution via discriminative classifier.
    Trains a classifier to distinguish baseline (0) vs current (1) data.
    Features that the classifier relies on most are the ones driving the drift.
    Returns: {feature_name: mean_abs_shap_value} sorted by importance.
    """
    # Build a combined dataset with labels
    baseline_subset = baseline_df[feature_names].copy()
    current_subset = current_df[feature_names].copy()

    baseline_subset["__label__"] = 0
    current_subset["__label__"] = 1

    combined = pd.concat([baseline_subset, current_subset], ignore_index=True)
    combined = combined.dropna()

    X = combined[feature_names]
    y = combined["__label__"]

    # Train a simple classifier to distinguish baseline vs current
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y)

    # Use SHAP TreeExplainer to see which features the classifier relied on
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # shap_values shape depends on sklearn/shap version — handle both cases
    if isinstance(shap_values, list):
        # older shap versions return a list [class_0_values, class_1_values]
        values_for_class_1 = shap_values[1]
    else:
        # newer shap versions return array of shape (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:
            values_for_class_1 = shap_values[:, :, 1]
        else:
            values_for_class_1 = shap_values

    mean_abs_shap = np.abs(values_for_class_1).mean(axis=0)

    feature_importance = {
        feature_names[i]: float(round(mean_abs_shap[i], 4))
        for i in range(len(feature_names))
    }

    # Sort by importance, descending
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Also compute classifier accuracy — if it's near 0.5, baseline and current are indistinguishable (no real drift)
    # if it's near 1.0, the classifier easily tells them apart (strong drift)
    accuracy = clf.score(X, y)

    return {
        "feature_importance": sorted_importance,
        "classifier_accuracy": round(float(accuracy), 4),
        "top_drift_driver": list(sorted_importance.keys())[0] if sorted_importance else None
    }
    