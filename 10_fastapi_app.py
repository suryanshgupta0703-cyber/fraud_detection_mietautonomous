"""
STEP 10: FASTAPI WEB SERVICE
=============================
Exposes the trained fraud detection model as a REST API.

Why FastAPI?
- Industry-standard way to deploy ML models
- Any application (mobile app, website, another program) can call it
- Auto-generates interactive API documentation (Swagger UI)
- Fast, modern, and easy to use

Run with: uvicorn 10_fastapi_app:app --reload
Then open: http://127.0.0.1:8000/docs  (interactive API docs)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: CREATE THE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A REST API that predicts whether a credit card transaction is genuine or fraudulent, using a trained Random Forest model.",
    version="1.0.0"
)

# ============================================================
# STEP 2: LOAD MODEL AND SCALER (runs once when server starts)
# ============================================================

try:
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except FileNotFoundError:
    model = None
    scaler = None
    print("❌ ERROR: Model files not found!")
    print("   Please run steps 1-6 first to train and save the model.")


# ============================================================
# STEP 3: DEFINE INPUT DATA FORMAT (Pydantic Model)
# ============================================================
# This tells FastAPI exactly what data to expect,
# validates it automatically, and shows it in the docs.

class Transaction(BaseModel):
    amount: float = Field(..., gt=0, le=100000, description="Transaction amount in ₹", example=5000)
    time: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)", example=14)
    merchant_type: int = Field(..., ge=1, le=5, description="Merchant type (1-5)", example=2)
    card_type: int = Field(..., ge=0, le=2, description="0=Debit, 1=Credit, 2=Prepaid", example=1)
    international: int = Field(..., ge=0, le=1, description="0=Domestic, 1=International", example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 5000,
                "time": 14,
                "merchant_type": 2,
                "card_type": 1,
                "international": 0
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    genuine_probability: float
    risk_level: str
    confidence: float


# ============================================================
# STEP 4: DEFINE API ENDPOINTS (ROUTES)
# ============================================================

@app.get("/")
def home():
    """
    Root endpoint - basic health check
    """
    return {
        "message": "💳 Credit Card Fraud Detection API is running!",
        "status": "active",
        "docs": "Visit /docs for interactive API documentation",
        "endpoints": {
            "/predict": "POST - Predict fraud for a single transaction",
            "/health": "GET - Check API and model status"
        }
    }


@app.get("/health")
def health_check():
    """
    Check if the API and model are working properly
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training steps first.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.

    Send transaction details (amount, time, merchant_type, card_type, international)
    and receive a prediction with fraud probability.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please run steps 1-6 to train and save the model first."
        )

    # Convert incoming request into a DataFrame (same column order as training)
    input_df = pd.DataFrame([{
        "amount": transaction.amount,
        "time": transaction.time,
        "merchant_type": transaction.merchant_type,
        "card_type": transaction.card_type,
        "international": transaction.international
    }])

    # Scale the input using the SAME scaler used during training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]                  # 0 or 1
    probabilities = model.predict_proba(input_scaled)[0]         # [genuine_prob, fraud_prob]

    fraud_prob = float(probabilities[1])
    genuine_prob = float(probabilities[0])

    # Determine risk level (same logic as your Streamlit app)
    if fraud_prob > 0.6:
        risk_level = "HIGH RISK - Likely Fraudulent"
        result = "FRAUD"
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM RISK - Suspicious"
        result = "SUSPICIOUS"
    else:
        risk_level = "LOW RISK - Likely Genuine"
        result = "GENUINE"

    return PredictionResponse(
        prediction=result,
        fraud_probability=round(fraud_prob, 4),
        genuine_probability=round(genuine_prob, 4),
        risk_level=risk_level,
        confidence=round(max(fraud_prob, genuine_prob), 4)
    )


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    """
    Predict fraud for MULTIPLE transactions at once.
    Send a list of transactions, get back a list of predictions.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    results = []
    for txn in transactions:
        input_df = pd.DataFrame([{
            "amount": txn.amount,
            "time": txn.time,
            "merchant_type": txn.merchant_type,
            "card_type": txn.card_type,
            "international": txn.international
        }])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        fraud_prob = float(model.predict_proba(input_scaled)[0][1])

        results.append({
            "input": txn.dict(),
            "prediction": "FRAUD" if prediction == 1 else "GENUINE",
            "fraud_probability": round(fraud_prob, 4)
        })

    return {"count": len(results), "results": results}


# ============================================================
# STEP 5: RUN THE SERVER (if this file is run directly)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI server...")
    print("   API will run at: http://127.0.0.1:8000")
    print("   Interactive docs: http://127.0.0.1:8000/docs")
    print("   Alternative docs: http://127.0.0.1:8000/redoc\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)