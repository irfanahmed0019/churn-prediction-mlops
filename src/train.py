"""Train and compare models with stratified CV, pick the best on CV PR-AUC,
evaluate once on a held-out test set, and save model + metrics + plots."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, PrecisionRecallDisplay,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load, CAT

SEED = 42


def make_pre(X, scale):
    num = [c for c in X.columns if c not in CAT]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])


def candidates(X):
    return {
        "dummy_most_frequent": Pipeline([("pre", make_pre(X, False)), ("m", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([("pre", make_pre(X, True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "random_forest": Pipeline([("pre", make_pre(X, False)), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=2))]),
        "hist_gradient_boosting": Pipeline([("pre", make_pre(X, False)), ("m", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, early_stopping=True, random_state=SEED))]),
    }


def main():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_res, models = {}, candidates(X)
    for name, p in models.items():
        r = cross_validate(p, Xtr, ytr, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"})
        cv_res[name] = {k: round(float(r[f"test_{k}"].mean()), 4) for k in ("roc_auc", "pr_auc", "f1")}
        print(name, cv_res[name])
    best = max((n for n in cv_res if n != "dummy_most_frequent"), key=lambda n: cv_res[n]["pr_auc"])
    model = models[best].fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    # choose threshold on training data via CV out-of-fold predictions (no test leakage)
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(models[best], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    pr, rc, th = precision_recall_curve(ytr, oof)
    f1s = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
    thr = float(th[f1s.argmax()])
    pred = (proba >= thr).astype(int)
    test = {
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "pr_auc": round(float(average_precision_score(yte, proba)), 4),
        "threshold": round(thr, 3),
        "precision": round(float(precision_score(yte, pred)), 4),
        "recall": round(float(recall_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "test_positive_rate": round(float(yte.mean()), 4),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    print("best", best, test)
    joblib.dump({"model": model, "threshold": thr, "columns": list(X.columns), "name": best}, "models/churn_model.joblib")
    json.dump({"cv_5fold_on_train": cv_res, "selected_model": best, "holdout_test": test}, open("reports/metrics.json", "w"), indent=2)
    PrecisionRecallDisplay.from_predictions(yte, proba, name=best)
    plt.title("Precision-recall on held-out test set"); plt.savefig("reports/pr_curve.svg", bbox_inches="tight"); plt.close()
    names = list(cv_res); vals = [cv_res[n]["pr_auc"] for n in names]
    plt.barh(names, vals); plt.xlabel("CV PR-AUC (5-fold, train split)"); plt.savefig("reports/model_comparison.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
