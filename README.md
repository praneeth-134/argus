# argus
# Argus — ML Model Monitoring Platform

Argus lets you register any ML model and monitor its predictions in production. Built after noticing that my Deep RL trading agent (AI Trader) degraded silently over time with no way to detect when or why.

## Why Argus

Models that work well in training often degrade in production due to data drift — the real world stops matching the data the model was trained on. Argus solves this by logging every prediction and (in upcoming versions) automatically detecting drift using statistical tests, then using an LLM to explain *why* the drift happened.

## Tech Stack

- **Backend:** FastAPI
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Cache:** Redis (real-time prediction storage)
- **Testing:** pytest

## Features (current)

- Register any ML model with versioning and feature metadata
- Log predictions in real-time via a simple Python SDK
- Query recent predictions per model
- Fully tested API endpoints

## Running locally

\`\`\`bash
# 1. Clone the repo
git clone https://github.com/praneeth-134/argus.git
cd argus

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env with your DATABASE_URL
# 5. Create tables
python create_tables.py

# 6. Run the server
uvicorn api.main:app --reload
\`\`\`

Visit `http://127.0.0.1:8000/docs` for the interactive API reference.

## Using the SDK

\`\`\`python
from sdk.argus_sdk import ArgusClient

client = ArgusClient()
client.log_prediction(
    model_id=1,
    input_features={"price": 182.5},
    prediction="BUY",
    confidence=0.91
)
\`\`\`

## Roadmap

- [ ] Drift detection engine (PSI, KS test) using Evidently AI
- [ ] SHAP-based feature importance on drift
- [ ] LLM diagnostic agent (Groq + Llama-3) for human-readable drift explanations
- [ ] Streamlit dashboard
- [ ] Docker + CI/CD deployment
