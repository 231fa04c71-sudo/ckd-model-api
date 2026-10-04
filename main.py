from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import time

# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CKD Prediction API",
    description="REST API for Chronic Kidney Disease prediction",
    version="1.0.0"
)

# ============================================================
# LOAD TRAINED MODEL AND FEATURE NAMES
# ============================================================

model = joblib.load("ckd_model.joblib")
feature_names = joblib.load("feature_names.joblib")

# ============================================================
# API REQUEST COUNTERS
# ============================================================

total_requests = 0
successful_predictions = 0
failed_requests = 0


# ============================================================
# GET /
# ============================================================

@app.get("/")
def root():
    return {
        "message": "CKD Prediction API is running",
        "version": "1.0.0"
    }


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# GET /model-info
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "model_name": "CKD Random Forest Classifier",
        "version": "1.0.0",
        "algorithm": "Random Forest",
        "number_of_features": len(feature_names),
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0
    }


# ============================================================
# GET /metrics
# ============================================================

@app.get("/metrics")
def metrics():
    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "failed_requests": failed_requests
    }


# ============================================================
# INPUT SCHEMA FOR /predict
# ============================================================

class CKDInput(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wbcc: float
    rbcc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
def predict(data: CKDInput):

    global total_requests
    global successful_predictions
    global failed_requests

    total_requests += 1

    start_time = time.time()

    try:

        # Convert API input into DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # Categorical columns used during training
        categorical_columns = [
            "rbc",
            "pc",
            "pcc",
            "ba",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane"
        ]

        # Clean categorical values
        for column in categorical_columns:
            input_data[column] = (
                input_data[column]
                .astype(str)
                .str.strip()
            )

        # One-hot encoding
        input_encoded = pd.get_dummies(
            input_data,
            columns=categorical_columns,
            dtype=int
        )

        # Match the exact feature columns used during training
        input_encoded = input_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Make prediction
        prediction = model.predict(input_encoded)[0]

        # Get prediction probability
        probability = model.predict_proba(input_encoded)[0].max()

        successful_predictions += 1

        # Convert numerical prediction to readable result
        if prediction == 1:
            result = "CKD"
        else:
            result = "Not CKD"

        response_time = time.time() - start_time

        return {
            "prediction": result,
            "model_output": int(prediction),
            "probability": round(float(probability), 4),
            "response_time_seconds": round(response_time, 4)
        }

    except Exception as e:

        failed_requests += 1

        return {
            "error": "Prediction failed",
            "details": str(e)
        }


# ============================================================
# GET /ready
# ============================================================

@app.get("/ready")
def ready():
    return {
        "ready": True,
        "model_loaded": model is not None
    }