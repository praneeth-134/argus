from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from database.connection import get_db
from database.models import MLModel, PredictionLog, DriftReport
from drift.detector import compute_drift
from drift.explainer import explain_drift
from llm.diagnostics import generate_diagnosis

router = APIRouter()


@router.post("/drift_check/{model_id}")
def run_drift_check(model_id: int, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.baseline_data:
        raise HTTPException(status_code=400, detail="No baseline data registered for this model")

    recent_logs = db.query(PredictionLog).filter(PredictionLog.model_id == model_id).all()
    if len(recent_logs) < 5:
        raise HTTPException(status_code=400, detail="Not enough recent predictions to compute drift (need at least 5)")

    current_rows = [log.input_features for log in recent_logs if log.input_features]
    current_df = pd.DataFrame(current_rows)
    baseline_df = pd.DataFrame(model.baseline_data)

    drift_results = compute_drift(baseline_df, current_df, model.feature_names)
    any_drifted = any(v["drifted"] for v in drift_results.values())

    shap_explanation = None
    llm_diagnosis = None
    if any_drifted:
        shap_explanation = explain_drift(baseline_df, current_df, model.feature_names)
        llm_diagnosis = generate_diagnosis(model.name, drift_results, shap_explanation)

    report = DriftReport(
        model_id=model_id,
        drift_scores=drift_results,
        drifted="yes" if any_drifted else "no"
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "drift_report_id": report.id,
        "model_id": model_id,
        "drifted": any_drifted,
        "details": drift_results,
        "shap_explanation": shap_explanation,
        "llm_diagnosis": llm_diagnosis
    }


@router.get("/drift_reports/{model_id}")
def get_drift_reports(model_id: int, db: Session = Depends(get_db)):
    reports = db.query(DriftReport).filter(DriftReport.model_id == model_id).order_by(DriftReport.created_at.desc()).all()
    return reports