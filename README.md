# Fraud Detection MLOps

An end-to-end fraud detection system demonstrating practical Machine Learning and MLOps engineering practices, from model training and threshold optimization to API deployment and production monitoring.

## 🚀 Project Overview

This project simulates a real-world fraud detection pipeline.

The system:

1. Generates transaction data
2. Trains a Random Forest classifier
3. Handles class imbalance
4. Evaluates model performance
5. Optimizes the fraud decision threshold
6. Packages and versions the trained model
7. Serves predictions through FastAPI
8. Validates API inputs
9. Monitors prediction behavior
10. Detects data drift
11. Provides automated API tests
12. Supports Docker deployment

## 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │ Transaction Data │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Random Forest ML │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
          ┌──────────────┐      ┌───────────────┐
          │ Model        │      │ Threshold     │
          │ Evaluation   │      │ Optimization  │
          └──────┬───────┘      └───────┬───────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                   ┌─────────────────┐
                   │ Model Packaging │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ FastAPI Service │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Predictions     │
                   └────────┬────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ Drift        │    │ Prediction   │
          │ Detection    │    │ Monitoring   │
          └──────────────┘    └──────────────┘

## Machine Learning

### Model

- Random Forest Classifier
- 200 trees
- `class_weight="balanced"`
- Probability-based fraud classification
- Optimized decision threshold

The model uses:

Transaction amount
Transaction hour
Customer age
Previous transaction count
Account age
Device risk score
Location risk score
Why threshold optimization?
Fraud detection is an imbalanced classification problem.
Accuracy alone can be misleading because legitimate transactions greatly outnumber fraudulent transactions.

Therefore, the project evaluates:

Precision
Recall
F1 score
ROC-AUC
Average Precision

The classification threshold is optimized using cross-validation rather than relying only on the default 0.5 threshold.

Note: The dataset is synthetic and the fraud labels are generated from predefined rules. Therefore, the evaluation results should be treated as a demonstration of the MLOps workflow rather than evidence of production fraud-detection performance.

🔍 MLOps Components
Model Packaging
The trained model and inference configuration are serialized using Joblib.
The model package contains:
model
threshold
features
Model Metadata

model_metadata.json stores information such as:

Model type
Number of estimators
Class weighting
Decision threshold
Feature list
Creation timestamp
Model Registry

model_registry.json demonstrates a lightweight model registry concept:

Production Model
      ↓
     v1
      ↓
fraud_detection_model.pkl

Experiment Tracking
experiment_tracking.csv records model configuration and experiment information.

Data Drift Detection
The project uses the Kolmogorov-Smirnov test to compare training and production-like feature distributions.

Example production drift simulation:
production_data["amount"] = production_data["amount"] * 1.8
production_data["device_risk_score"] = (
    production_data["device_risk_score"] + 0.15
).clip(0, 1)

The system can identify significant distribution changes and generate a drift alert.

🌐 FastAPI

The trained model is exposed through a REST API.

Health Check
GET /health

Example response:

{
  "status": "healthy"
}
Fraud Prediction
POST /predict

Example request:

{
  "amount": 100,
  "transaction_hour": 14,
  "customer_age": 35,
  "previous_transactions": 50,
  "account_age_days": 500,
  "device_risk_score": 0.2,
  "location_risk_score": 0.1
}

The API returns the fraud probability and classification.

Input Validation

FastAPI/Pydantic validation prevents invalid values such as:

Transaction hour outside 0–23
Risk scores outside 0–1
Invalid customer age
Non-positive transaction amounts

🧪 Testing

Automated API tests are provided using FastAPI's TestClient.

Tests cover:

Root endpoint
Health endpoint
Prediction endpoint

Run: pytest

🐳 Docker

The application can be containerized using Docker.

Build:

docker build -t fraud-detection-api .

Run:

docker run -p 8000:8000 fraud-detection-api

The API will then be available at: http://localhost:8000


📁 Project Structure
fraud-detection-mlops/
│
├── app.py
├── fraud_detection_model.pkl
├── model_metadata.json
├── model_registry.json
├── experiment_tracking.csv
├── environment_versions.txt
│
├── test_app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md


🛠️ Technology Stack
Area	Technology
Language	Python
ML	Scikit-learn
Model	Random Forest
API	FastAPI
Validation	Pydantic
Testing	Pytest
Serialization	Joblib
Data	Pandas / NumPy
Statistics	SciPy
Containerization	Docker
Version Control	Git / GitHub

🔄 End-to-End Workflow

Data
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Cross-Validation
 ↓
Model Evaluation
 ↓
Threshold Optimization
 ↓
Model Packaging
 ↓
Model Versioning
 ↓
FastAPI Deployment
 ↓
Automated Testing
 ↓
Production Monitoring
 ↓
Drift Detection

💡 Key Engineering Decisions
Why Random Forest?
Random Forest provides a strong baseline for tabular fraud data and can model nonlinear relationships without requiring extensive feature transformations.

Why optimize the threshold?
The default classification threshold is not necessarily appropriate for fraud detection.
A lower threshold can increase fraud recall but may also increase false positives.
The appropriate operating point depends on the business cost of:

False Positive
vs.
False Negative

Why monitor drift?
A model can perform well during development but encounter a different feature distribution in production.
Monitoring helps identify changes that may require investigation, retraining, or changes to the decision process.

⚠️ Limitations

This project is designed as an educational and portfolio demonstration.
The dataset is synthetic and the fraud labels are generated using artificial rules.
A production system would additionally require:

Real transaction data
Robust feature engineering
Feature stores
Real-time data pipelines
Model performance monitoring with delayed labels
Automated retraining
Model governance
Authentication and authorization
Secure secrets management
CI/CD
Cloud deployment
Scalable infrastructure
Business-cost-based threshold selection


🎯 What This Project Demonstrates

This project demonstrates practical experience across the ML lifecycle:

Machine Learning
       +
Software Engineering
        +
API Development
        +
Testing
        +
Model Deployment
        +
Monitoring
        +
MLOps

The focus is not only on training a model, but on building the surrounding engineering system required to operate a machine-learning model as a service.
