"""Load the OpenML churn dataset (id 40701, 5,000 telecom customers)."""
import pandas as pd
from sklearn.datasets import fetch_openml

DROP = ["phone_number"]  # unique identifier, not a feature
CAT = ["state", "area_code", "international_plan", "voice_mail_plan"]


def load():
    d = fetch_openml(data_id=40701, as_frame=True, parser="auto")
    X = d.data.drop(columns=DROP).copy()
    for c in CAT:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in CAT]
    X[num] = X[num].astype(float)
    y = d.target.astype(int)
    return X, y
