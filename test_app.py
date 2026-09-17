from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    transaction = {
        "amount": 100,
        "transaction_hour": 14,
        "customer_age": 35,
        "previous_transactions": 50,
        "account_age_days": 500,
        "device_risk_score": 0.2,
        "location_risk_score": 0.1
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200
    assert "fraud_probability" in response.json()
    assert "is_fraud" in response.json()
