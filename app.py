"""
app.py
Flask REST API for the IoT Security Monitor.

Endpoints
─────────
GET  /devices            → list all simulated devices
POST /analyze            → classify a single traffic sample
GET  /simulate           → generate & classify a live random sample
GET  /history            → return recent prediction history (last 50)
GET  /stats              → aggregate alert stats per device
"""

import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from flask import Flask, jsonify, request
from datetime import datetime
from collections import deque

# Local modules
from ml_model.anomaly_detector import predict, load_model
from simulation.device_simulator import DEVICE_PROFILES

app = Flask(__name__)

# ── In-memory ring buffer for recent events ───────────────────────────────────
HISTORY: deque = deque(maxlen=50)

# ── Pre-load model at startup ────────────────────────────────────────────────
load_model()


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/devices", methods=["GET"])
def list_devices():
    """Return all known device names."""
    return jsonify({"devices": list(DEVICE_PROFILES.keys())})


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Classify a traffic sample.
    Body (JSON): { "device": str, "packet_count": float,
                   "request_freq": float, "data_size_kb": float }
    """
    data = request.get_json(force=True)
    required = {"device", "packet_count", "request_freq", "data_size_kb"}
    missing  = required - data.keys()
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = predict(data["packet_count"], data["request_freq"], data["data_size_kb"])
    event  = {
        "timestamp":    datetime.utcnow().isoformat(),
        "device":       data["device"],
        "packet_count": data["packet_count"],
        "request_freq": data["request_freq"],
        "data_size_kb": data["data_size_kb"],
        **result,
    }
    HISTORY.appendleft(event)

    # Console alert
    tag = "🔴 ALERT" if result["anomaly"] else "🟢 OK   "
    print(f"{tag} [{event['timestamp']}] {data['device']:15s} → {result['status']}")

    return jsonify(event)


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/simulate", methods=["GET"])
def simulate():
    """Generate a random live sample (80 % normal, 20 % anomalous) and classify."""
    device  = random.choice(list(DEVICE_PROFILES.keys()))
    profile = DEVICE_PROFILES[device]
    anomaly = random.random() < 0.20   # 20 % chance of spike

    if anomaly:
        mult = random.uniform(3, 6)
        pkt  = profile["packet_count"][0] * mult
        freq = profile["request_freq"][0]  * mult
        size = profile["data_size_kb"][0]  * mult
    else:
        pkt  = max(0, np.random.normal(*profile["packet_count"]))
        freq = max(0, np.random.normal(*profile["request_freq"]))
        size = max(0, np.random.normal(*profile["data_size_kb"]))

    result = predict(pkt, freq, size)
    event  = {
        "timestamp":    datetime.utcnow().isoformat(),
        "device":       device,
        "packet_count": round(pkt,  2),
        "request_freq": round(freq, 2),
        "data_size_kb": round(size, 2),
        **result,
    }
    HISTORY.appendleft(event)

    tag = "🔴 ALERT" if result["anomaly"] else "🟢 OK   "
    print(f"{tag} [LIVE] {device:15s} → {result['status']}  score={result['score']:.4f}")

    return jsonify(event)


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    """Return up to 50 recent events."""
    return jsonify(list(HISTORY))


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    """Aggregate normal vs suspicious counts per device."""
    totals = {}
    for event in HISTORY:
        d = event["device"]
        totals.setdefault(d, {"total": 0, "alerts": 0})
        totals[d]["total"]  += 1
        totals[d]["alerts"] += 1 if event["anomaly"] else 0
    return jsonify(totals)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting IoT Security Monitor API on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
