"""
Credit Risk Prediction API - FastAPI Backend
------------------------------------------------
Trained model (credit_risk_model.pkl) load karke risk-score endpoint serve karta hai.

RUN:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8001

ENDPOINTS:
    GET  /                       -> health check
    POST /predict-risk           -> single customer ka risk score
    POST /predict-risk-bulk      -> pura customer_data.csv analyze karke sab customers ka risk
    GET  /customers               -> saved customer list with risk scores
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Credit Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "credit_risk_model.pkl"
DATA_PATH = "customer_data.csv"

model_bundle = None


class CustomerInput(BaseModel):
    customer_id: str = "unknown"
    total_purchases: int
    total_credit_amount: float
    total_paid_amount: float
    avg_days_to_pay: int
    max_days_overdue: int


@app.on_event("startup")
def load_model():
    global model_bundle
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  credit_risk_model.pkl nahi mila. Pehle train_model.py run karein.")


@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model_bundle is not None
    }


def _risk_label(prob):
    if prob >= 0.6:
        return "High Risk"
    elif prob >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"


def _predict_one(row: dict):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    credit_utilization = row["total_paid_amount"] / (row["total_credit_amount"] or 1)
    X = pd.DataFrame([{
        "total_purchases": row["total_purchases"],
        "total_credit_amount": row["total_credit_amount"],
        "total_paid_amount": row["total_paid_amount"],
        "avg_days_to_pay": row["avg_days_to_pay"],
        "max_days_overdue": row["max_days_overdue"],
        "credit_utilization": credit_utilization
    }])[features]

    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0][1])
    return {
        "customer_id": row.get("customer_id", "unknown"),
        "risk_probability": round(prob, 4),
        "risk_label": _risk_label(prob)
    }


@app.post("/predict-risk")
def predict_risk(customer: CustomerInput):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model load nahi hua. Pehle train_model.py run karein."
        )
    return _predict_one(customer.dict())


@app.get("/customers")
def get_customers(limit: int = 50):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model load nahi hua.")
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="customer_data.csv nahi mila")

    df = pd.read_csv(DATA_PATH).head(limit)
    results = []
    for _, row in df.iterrows():
        results.append(_predict_one(row.to_dict()))

    # Sort: high risk pehle
    results.sort(key=lambda r: r["risk_probability"], reverse=True)
    return {"customers": results}
