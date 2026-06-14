"""Latih 3 model (Random Forest, Logistic Regression, Gradient Boosting),
simpan ketiganya ke folder models/, dan tulis metrik nyata ke models/metrics.json.

Pakai dataset CDC Diabetes Health Indicators (balanced 50/50) yang sama dengan notebook.
Jalankan: python -m scripts.train_models  (atau: python scripts/train_models.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "cdc_diabetes_health_indicators.csv"
MODELS_DIR = BASE_DIR / "models"

# key -> (label tampilan, file pkl, konstruktor)
MODEL_SPECS = {
    "gb": ("Gradient Boosting", "model_gb.pkl", lambda: GradientBoostingClassifier(random_state=42)),
    # max_depth/min_samples_leaf membatasi ukuran pickle (RF default bisa ratusan MB)
    # tanpa menurunkan akurasi secara berarti.
    "rf": ("Random Forest", "model_rf.pkl", lambda: RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=25, random_state=42, n_jobs=-1)),
    "lr": ("Logistic Regression", "model_lr.pkl", lambda: LogisticRegression(max_iter=1000, random_state=42)),
}
# Model utama yang dipilih sebagai "model_diabetes_terbaik.pkl" (3 jadi 1)
BEST_KEY = "gb"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Letakkan cdc_diabetes_health_indicators.csv di root proyek."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna()
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    metrics = {}
    for key, (label, fname, make) in MODEL_SPECS.items():
        print(f"Melatih {label} ...")
        model = make()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics[key] = {
            "label": label,
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, average="weighted") * 100, 2),
            "recall": round(recall_score(y_test, y_pred, average="weighted") * 100, 2),
            "f1": round(f1_score(y_test, y_pred, average="weighted") * 100, 2),
        }
        joblib.dump(model, MODELS_DIR / fname)
        print(f"  -> tersimpan {fname} | {metrics[key]}")

    # Simpan juga model terbaik sebagai nama lama (kompatibilitas mundur)
    best_label, best_fname, _ = MODEL_SPECS[BEST_KEY]
    joblib.dump(joblib.load(MODELS_DIR / best_fname), MODELS_DIR / "model_diabetes_terbaik.pkl")

    payload = {"best": BEST_KEY, "models": metrics}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nModel terbaik: {best_label}. Metrik ditulis ke models/metrics.json")


if __name__ == "__main__":
    main()
