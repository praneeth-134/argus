DIAGNOSTIC_SYSTEM_PROMPT = """You are an expert ML monitoring analyst. You will be given statistical drift detection results for a machine learning model in production, including PSI scores, KS test results, and SHAP-based feature attribution.

Your job is to diagnose the drift in plain English and recommend specific actions.

Respond ONLY in valid JSON with this exact structure, no markdown formatting, no preamble:
{
  "summary": "one sentence describing what happened",
  "root_cause": "2-3 sentences explaining the likely cause based on the data given",
  "severity": "low" or "medium" or "high",
  "recommended_actions": ["action 1", "action 2", "action 3"]
}
"""


def build_diagnostic_prompt(model_name: str, drift_results: dict, shap_explanation: dict) -> str:
    drifted_features = [f for f, v in drift_results.items() if v["drifted"]]
    stable_features = [f for f, v in drift_results.items() if not v["drifted"]]

    prompt = f"Model: {model_name}\n\n"
    prompt += f"Drifted features: {', '.join(drifted_features) if drifted_features else 'none'}\n"
    prompt += f"Stable features: {', '.join(stable_features) if stable_features else 'none'}\n\n"

    prompt += "Detailed drift scores:\n"
    for feature, scores in drift_results.items():
        prompt += f"- {feature}: PSI={scores['psi']}, KS p-value={scores['p_value']}, drifted={scores['drifted']}\n"

    if shap_explanation:
        prompt += f"\nSHAP feature importance (which features the model relies on to distinguish baseline from current data):\n"
        for feature, importance in shap_explanation["feature_importance"].items():
            prompt += f"- {feature}: {importance}\n"
        prompt += f"\nClassifier accuracy distinguishing baseline vs current: {shap_explanation['classifier_accuracy']}\n"
        prompt += f"Top drift driver: {shap_explanation['top_drift_driver']}\n"

    prompt += "\nDiagnose this drift and recommend actions."
    return prompt