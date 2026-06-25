from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from database.connection import Base

class MLModel(Base):
    __tablename__ = "ml_models"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    description = Column(String)
    feature_names = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    input_features = Column(JSON)
    prediction = Column(Float)
    confidence = Column(Float)
    logged_at = Column(DateTime, default=datetime.utcnow)

class DriftReport(Base):
    __tablename__ = "drift_reports"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    drift_scores = Column(JSON)
    drifted = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)