# 🌐 AI-Based Micro-Zone Disaster Risk Scoring System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, end-to-end geospatial machine learning system designed to compute, explain, and serve hyper-local disaster vulnerability scores (0–100) across micro-zones. The platform integrates real-world historical disaster telemetry, geospatial feature engineering, gradient boosting algorithms, local SHAP explainability, automated drift monitoring, and production-ready REST APIs.

---

## 📌 Architecture Overview

The system is designed with a multi-stage modular pipeline:
1. **Data Ingestion Layer**: Aggregates real-world historical disaster telemetry (`data/disasterIND.csv`) with spatial static terrain, demographic, and infrastructure indices.
2. **Preprocessing & Quality Assurance**: Outlier trimming using Interquartile Range (IQR) and multicollinearity elimination via Variance Inflation Factor (VIF) filtering.
3. **Feature Engineering**: Derives micro-zone vulnerability indicators, interaction terms, and temporal recurrence metrics.
4. **Model Training & Optimization**: Trains tuned **XGBoost Regressors** and **Logistic Regression** classifiers with cross-validation.
5. **Explainability Engine (XAI)**: Uses **SHAP TreeExplainer** to quantify feature attribution and pinpoint root causes of elevated risk.
6. **API Microservice**: High-concurrency **FastAPI** service serving low-latency inference and on-the-fly local explanations.
7. **Drift Detection & Recalibration**: Continuous monitoring using Population Stability Index (PSI) to trigger automated retraining when data drifts.

---

## 🚀 Key Features

- **Micro-Zone Spatial Discretization**: Dynamically projects geographical coordinates into UTM CRS for consistent metric distance calculations.
- **Multicollinearity-Robust Pipeline**: Computes dynamic VIF matrices to eliminate redundant or highly correlated features prior to model fit.
- **Continuous 0–100 Risk Scoring**: Maps multi-factor disaster threats into an intuitive, normalized score with defined risk categories:
  - 🟢 **Low** (`0.0 - 30.0`)
  - 🟡 **Moderate** (`30.1 - 60.0`)
  - 🟠 **High** (`60.1 - 80.0`)
  - 🔴 **Critical** (`80.1 - 100.0`)
- **Interpretability & Explainability**: Integrates SHAP background distributions to decompose risk scores into top-contributing local environmental and demographic drivers.
- **Real-Time Data Drift Detection**: Implements Population Stability Index (PSI) calculations across incoming feature batches to detect distribution shifts and signal retraining triggers.
- **Interactive Visualizations**: Generates Plotly geospatial choropleths, Folium spatial map overlays, and diagnostic ML curves.

---

## 📁 Repository Structure

```text
Micro-Zone-Disaster-Risk-Scoring-System/
├── api/
│   ├── app.py             # Production FastAPI application & inference endpoints
│   └── README.md          # API quickstart and invocation guide
├── data/
│   ├── disasterIND.csv    # Historical disaster event dataset
│   └── zone_year_raw.parquet # Synthesized micro-zone geospatial data
├── models/
│   └── risk_model_latest.joblib # Trained production model bundle (pipeline + SHAP)
├── notebooks/
│   └── micro_zone_disaster_risk_scoring_system.ipynb # End-to-end research & training notebook
├── gg.excalidraw.png      # High-level architecture and workflow diagram
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules for Python, models, and environments
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/DigarSingh/Micro-Zone-Disaster-Risk-Scoring-System.git
cd Micro-Zone-Disaster-Risk-Scoring-System
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Model Training & Notebook Execution

To explore the data, inspect the EDA, re-train models, or generate evaluation figures:

```bash
jupyter lab
```

Open `notebooks/micro_zone_disaster_risk_scoring_system.ipynb` and run all cells. Running the notebook will:
- Process raw disaster data and perform spatial grid discretization.
- Perform feature engineering, outlier clipping, and VIF reduction.
- Tune and evaluate XGBoost models.
- Package the trained pipeline and SHAP background data into `models/risk_model_latest.joblib`.

---

## 🌐 FastAPI Service

The FastAPI service exposes real-time inference and explanation endpoints loaded directly from the model artifact bundle.

### Start the Server
```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Access the interactive OpenAPI / Swagger documentation at:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### API Endpoints

#### 1. Predict Zone Risk
- **Endpoint**: `POST /predict_zone_risk`
- **Request Body**:
```json
{
  "zone_id": "ZONE_MUMBAI_042",
  "features": {
    "elevation_m": 12.5,
    "slope_deg": 1.2,
    "drainage_density": 0.85,
    "soil_permeability": 0.22,
    "historical_disaster_count": 8,
    "avg_annual_rainfall_mm": 2450.0,
    "population_density": 18500.0,
    "critical_infrastructure_count": 14,
    "hospital_distance_km": 3.8
  }
}
```
- **Response**:
```json
{
  "zone_id": "ZONE_MUMBAI_042",
  "risk_score": 78.4,
  "risk_category": "High",
  "model_version": "20260218_155446",
  "trained_at_utc": "2026-02-18T15:54:46"
}
```

#### 2. Explain Zone Risk (SHAP Attributions)
- **Endpoint**: `POST /explain_zone`
- **Request Body**: (Same format as above)
- **Response**:
```json
{
  "zone_id": "ZONE_MUMBAI_042",
  "top_contributing_features": [
    { "feature": "avg_annual_rainfall_mm", "shap_value": 14.21, "abs_shap": 14.21 },
    { "feature": "population_density", "shap_value": 9.85, "abs_shap": 9.85 },
    { "feature": "elevation_m", "shap_value": -6.40, "abs_shap": 6.40 },
    { "feature": "drainage_density", "shap_value": -4.12, "abs_shap": 4.12 }
  ],
  "model_version": "20260218_155446"
}
```

---

## 📊 Risk Score Categories

| Score Range | Category | Description | Recommended Action |
|:---:|:---:|:---|:---|
| **0 – 30** | 🟢 **Low** | Minimal historical hazard exposure, favorable terrain, resilient infrastructure. | Routine monitoring & regular maintenance. |
| **31 – 60** | 🟡 **Moderate** | Moderate vulnerability during peak seasonal weather events. | Pre-monsoon clearing and resource staging. |
| **61 – 80** | 🟠 **High** | High probability of severe impact, dense population or vulnerable elevation. | Early warning activation, rapid response deployment. |
| **81 – 100** | 🔴 **Critical** | Extreme multi-hazard vulnerability, high casualty and infrastructure risk. | Immediate pre-emptive evacuation and targeted mitigation. |

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
