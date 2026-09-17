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
