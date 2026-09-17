import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Fraud Detection API",
    description="Machine learning API for fraud transaction prediction",
    version="1.0.0"
)


# Load model package
model_package = joblib.load("fraud_detection_model.pkl")

model = model_package["model"]
threshold = model_package["threshold"]
features = model_package["features"]


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    transaction_hour: int = Field(ge=0, le=23)
    customer_age: int = Field(ge=18, le=120)
    previous_transactions: int = Field(ge=0)
    account_age_days: int = Field(ge=0)
    device_risk_score: float = Field(ge=0, le=1)
    location_risk_score: float = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "threshold": float(threshold)
    }


@app.post("/predict")
def predict(transaction: Transaction):

    try:
        transaction_data = transaction.model_dump()

        transaction_df = pd.DataFrame([transaction_data])
        transaction_df = transaction_df[features]

        probability = model.predict_proba(transaction_df)[0, 1]

        prediction = int(probability >= threshold)

        return {
            "fraud_probability": float(probability),
            "is_fraud": bool(prediction),
            "result": (
                "Potential fraud"
                if prediction == 1
                else "Legitimate transaction"
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
