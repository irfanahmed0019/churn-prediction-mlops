from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
SAMPLE = dict(state="16", account_length=128, area_code="415", international_plan="0", voice_mail_plan="1",
              number_vmail_messages=25, total_day_minutes=265.1, total_day_calls=110, total_day_charge=45.07,
              total_eve_minutes=197.4, total_eve_calls=99, total_eve_charge=16.78, total_night_minutes=244.7,
              total_night_calls=91, total_night_charge=11.01, total_intl_minutes=10.0, total_intl_calls=3,
              total_intl_charge=2.7, number_customer_service_calls=1)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_bad_input_rejected():
    bad = dict(SAMPLE); del bad["account_length"]
    assert client.post("/predict", json=bad).status_code == 422
