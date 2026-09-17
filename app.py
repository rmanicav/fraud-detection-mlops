
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Fraud Detection API")

model_package = joblib.load("fraud_detection_model.pkl")

model = model_package["model"]
threshold = model_package["threshold"]
features = model_package["features"]


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: dict):
    transaction_df = pd.DataFrame([transaction])
    transaction_df = transaction_df[features]

    probability = model.predict_proba(transaction_df)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction,
        "result": (
            "Potential fraud"
            if prediction == 1
            else "Legitimate transaction"
        )
    }
