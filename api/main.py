from fastapi import FastAPI
from datetime import datetime
from api.routes import models

app = FastAPI(
    title="Argus — ML Monitoring Platform",
    description="Monitor your ML models in production",
    version="0.1.0"
)

app.include_router(models.router)

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }

@app.get("/")
def root():
    return {"message": "Welcome to Argus. Visit /docs for the API reference."}