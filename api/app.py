# Getaround Pricing Prediction API
# ---------------------------------------------------------------------------

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "getaround_pricing_model.joblib"
)


# ---------------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------------

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Getaround Pricing Prediction API",
    description=(
        "API for predicting the daily rental price of a vehicle "
        "from its characteristics and equipment."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Prediction schemas
# ---------------------------------------------------------------------------

class CarFeatures(BaseModel):
    model_key: str
    mileage: int = Field(ge=0)
    engine_power: int = Field(gt=0)
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


class PricePrediction(BaseModel):
    predicted_rental_price_per_day: float


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="API status",
    description="Check that the Getaround pricing API is running.",
)
def root():
    return {
        "message": "Getaround Pricing Prediction API",
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PricePrediction,
    summary="Predict daily rental price",
    description=(
        "Predict the daily rental price of one vehicle from its "
        "characteristics and equipment."
    ),
)
def predict(features: CarFeatures):
    input_data = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_rental_price_per_day": round(float(prediction), 2)
    }