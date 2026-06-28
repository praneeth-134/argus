from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from database.connection import get_db
from database.models import MLModel

router = APIRouter()

class ModelRegisterRequest(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    feature_names: List[str]
    baseline_data: Optional[Dict[str, List[float]]] = None

@router.post("/register_model")
def register_model(payload: ModelRegisterRequest, db: Session = Depends(get_db)):
    new_model = MLModel(
        name=payload.name,
        version=payload.version,
        description=payload.description,
        feature_names=payload.feature_names,
        baseline_data=payload.baseline_data
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return {"model_id": new_model.id, "message": "Model registered successfully"}

@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(MLModel).all()
    return models