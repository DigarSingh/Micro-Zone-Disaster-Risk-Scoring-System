
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ZoneFeatures(BaseModel):
    zone_id: str | None = None
    features: dict[str, float | int | None]


def map_score_to_category(score: float) -> str:
    if score <= 30:
        return 'Low'
    if score <= 60:
        return 'Moderate'
    if score <= 80:
        return 'High'
    return 'Critical'


def load_bundle(models_dir: str) -> dict:
    path = os.path.join(models_dir, 'risk_model_latest.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model not found at {path}')
    b = joblib.load(path)
    if not isinstance(b, dict):
        raise TypeError('Expected dict bundle artifact. Re-run training to regenerate artifacts.')
    return b


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
bundle = load_bundle(MODELS_DIR)

pipeline = bundle['pipeline']
feature_names = bundle['feature_names']
shap_background = bundle['shap_background']

app = FastAPI(title='Micro-Zone Disaster Risk Scoring API', version='1.0')


@app.get('/get_all_zones')
def get_all_zones():
    # Example endpoint: in production, query a database or feature store
    return {'message': 'Implement DB/feature-store lookup for zones here.'}


@app.post('/predict_zone_risk')
def predict_zone_risk(payload: ZoneFeatures):
    # Align to model schema
    row = {f: payload.features.get(f) for f in feature_names}
    X = pd.DataFrame([row], columns=feature_names)
    try:
        score = float(pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction error: {e}')

    score = float(np.clip(score, 0, 100))
    return {
        'zone_id': payload.zone_id,
        'risk_score': score,
        'risk_category': map_score_to_category(score),
        'model_version': bundle.get('version'),
        'trained_at_utc': bundle.get('trained_at_utc'),
    }


@app.post('/explain_zone')
def explain_zone(payload: ZoneFeatures):
    row = {f: payload.features.get(f) for f in feature_names}
    X = pd.DataFrame([row], columns=feature_names)

    try:
        X_proc = pipeline.named_steps['preprocess'].transform(X)
        explainer = shap.TreeExplainer(pipeline.named_steps['model'], data=shap_background)
        sv = explainer.shap_values(X_proc)[0]
        contrib = sorted(
            [{'feature': fn, 'shap_value': float(v), 'abs_shap': float(abs(v))} for fn, v in zip(feature_names, sv)],
            key=lambda d: d['abs_shap'],
            reverse=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Explain error: {e}')

    return {
        'zone_id': payload.zone_id,
        'top_contributing_features': contrib[:8],
        'model_version': bundle.get('version'),
    }