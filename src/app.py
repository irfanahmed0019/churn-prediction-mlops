"""FastAPI service that serves the trained churn model."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ART = joblib.load(Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib")
app = FastAPI(title="Churn prediction API", version="1.0")


class Customer(BaseModel):
    state: str = Field(examples=["16"])
    account_length: float
    area_code: str = Field(examples=["415"])
    international_plan: str = Field(examples=["0"])
    voice_mail_plan: str = Field(examples=["1"])
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: float
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: float
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: float
    total_intl_charge: float
    number_customer_service_calls: float


@app.get("/health")
def health():
    return {"status": "ok", "model": ART["name"]}


@app.post("/predict")
def predict(c: Customer):
    row = pd.DataFrame([c.model_dump()])[ART["columns"]]
    p = float(ART["model"].predict_proba(row)[0, 1])
    return {"churn_probability": round(p, 4), "will_churn": p >= ART["threshold"], "threshold": ART["threshold"]}
