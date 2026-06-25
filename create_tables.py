from database.connection import engine, Base
from database.models import MLModel, PredictionLog, DriftReport

Base.metadata.create_all(bind=engine)
print("Tables created successfully!")