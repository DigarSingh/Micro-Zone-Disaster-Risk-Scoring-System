# FastAPI Disaster Risk Scoring Service

This directory contains the production FastAPI inference microservice for the Micro-Zone Disaster Risk Scoring System.

## Prerequisites

Ensure dependencies are installed from the repository root:

```bash
pip install -r ../requirements.txt
```

Verify that the model bundle exists at `../models/risk_model_latest.joblib`. If not present, execute the training notebook in `../notebooks/`.

## Running the API

From the root directory:

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

## Interactive Documentation

Once the server is running, navigate to:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

