from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from database.connection import get_db
from database.models import MLModel, PredictionLog, DriftReport
from drift.detector import compute_drift

router = APIRouter()


@router.post("/drift_check/{model_id}")
def run_drift_check(model_id: int, db: Session = Depends(get_db)):
    # 1. Fetch the model and its baseline data
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.baseline_data:
        raise HTTPException(status_code=400, detail="No baseline data registered for this model")

    # 2. Fetch recent predictions for this model from PostgreSQL
    recent_logs = db.query(PredictionLog).filter(PredictionLog.model_id == model_id).all()
    if len(recent_logs) < 5:
        raise HTTPException(status_code=400, detail="Not enough recent predictions to compute drift (need at least 5)")

    # 3. Build current_df from input_features stored in each log
    current_rows = [log.input_features for log in recent_logs if log.input_features]
    current_df = pd.DataFrame(current_rows)

    # 4. Build baseline_df from stored baseline_data
    baseline_df = pd.DataFrame(model.baseline_data)

    # 5. Compute drift
    drift_results = compute_drift(baseline_df, current_df, model.feature_names)

    # 6. Determine overall drift status
    any_drifted = any(v["drifted"] for v in drift_results.values())

    # 7. Save the report
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
        "details": drift_results
    }


@router.get("/drift_reports/{model_id}")
def get_drift_reports(model_id: int, db: Session = Depends(get_db)):
    reports = db.query(DriftReport).filter(DriftReport.model_id == model_id).order_by(DriftReport.created_at.desc()).all()
    return reports