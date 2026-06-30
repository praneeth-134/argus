import json
from llm.groq_client import call_llm
from llm.prompts import DIAGNOSTIC_SYSTEM_PROMPT, build_diagnostic_prompt


def generate_diagnosis(model_name: str, drift_results: dict, shap_explanation: dict) -> dict:
    user_prompt = build_diagnostic_prompt(model_name, drift_results, shap_explanation)
    raw_response = call_llm(DIAGNOSTIC_SYSTEM_PROMPT, user_prompt)

    try:
        # Sometimes models wrap JSON in markdown code fences — strip those if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        diagnosis = json.loads(cleaned.strip())
        return diagnosis
    except json.JSONDecodeError:
        # fallback if the model didn't return clean JSON
        return {
            "summary": "Diagnosis generated but could not be parsed as structured JSON",
            "root_cause": raw_response,
            "severity": "unknown",
            "recommended_actions": []
        }