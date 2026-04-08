"""
anomaly_detector.py
Trains an Isolation Forest on normal IoT traffic and exposes a
predict() helper that returns "Normal" or "Suspicious".
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "ml_model/isolation_forest.pkl"
SCALER_PATH = "ml_model/scaler.pkl"
DATA_PATH   = "data/simulated_iot_data.csv"

FEATURES = ["packet_count", "request_freq", "data_size_kb"]


def train_model(data_path: str = DATA_PATH) -> tuple:
    """
    Train Isolation Forest on *normal* rows only (mimicking real-world
    scenarios where we only have clean baseline data).
    Returns (model, scaler).
    """
    df = pd.read_csv(data_path)
    normal_df = df[df["label"] == "normal"][FEATURES]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_df)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # expect ~5 % outliers in production stream
        random_state=42,
    )
    model.fit(X_train)

    os.makedirs("ml_model", exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    print(f"[ML] Model trained on {len(normal_df)} normal samples.")
    print(f"[ML] Saved → {MODEL_PATH}, {SCALER_PATH}")
    return model, scaler


def load_model() -> tuple:
    """Load persisted model and scaler; train if not found."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        return model, scaler
    print("[ML] No saved model found — training now …")
    return train_model()


def predict(packet_count: float, request_freq: float, data_size_kb: float) -> dict:
    """
    Classify a single observation.
    Returns {"status": "Normal"|"Suspicious", "score": float, "anomaly": bool}
    """
    model, scaler = load_model()
    X = scaler.transform([[packet_count, request_freq, data_size_kb]])
    raw_score = model.decision_function(X)[0]   # higher = more normal
    prediction = model.predict(X)[0]             # 1 = normal, -1 = anomaly
    return {
        "status":  "Normal" if prediction == 1 else "Suspicious",
        "score":   round(float(raw_score), 4),
        "anomaly": prediction == -1,
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, scaler = train_model()

    # Quick smoke-test
    test_cases = [
        ("Normal Camera",    120, 30, 500),
        ("Attack Spike",     720, 180, 3000),
        ("Normal Bulb",      20,  5,  10),
        ("Anomalous Sensor", 350, 90, 600),
    ]
    print("\n── Smoke-test results ───────────────────────────────")
    for label, p, r, d in test_cases:
        result = predict(p, r, d)
        tag = "🔴 ALERT" if result["anomaly"] else "🟢 OK"
        print(f"{tag}  {label:25s}  status={result['status']:12s}  score={result['score']:.4f}")
