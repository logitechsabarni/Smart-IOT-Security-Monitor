"""
device_simulator.py
Simulates IoT device traffic for three device types:
  - Smart Camera
  - Smart Bulb
  - Smart Sensor
Generates both normal and anomalous (attack-like) traffic and saves to CSV.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Device traffic profiles (mean, std) ──────────────────────────────────────
DEVICE_PROFILES = {
    "Smart Camera":  {"packet_count": (120, 15), "request_freq": (30, 5),  "data_size_kb": (500, 80)},
    "Smart Bulb":    {"packet_count": (20,  5),  "request_freq": (5,  2),  "data_size_kb": (10,  3)},
    "Smart Sensor":  {"packet_count": (50,  10), "request_freq": (15, 3),  "data_size_kb": (80,  15)},
}

def generate_normal_traffic(device: str, n_samples: int = 200) -> pd.DataFrame:
    """Return a DataFrame of *normal* traffic rows for one device."""
    profile = DEVICE_PROFILES[device]
    rows = []
    for _ in range(n_samples):
        rows.append({
            "device":        device,
            "packet_count":  max(0, np.random.normal(*profile["packet_count"])),
            "request_freq":  max(0, np.random.normal(*profile["request_freq"])),
            "data_size_kb":  max(0, np.random.normal(*profile["data_size_kb"])),
            "label":         "normal",
        })
    return pd.DataFrame(rows)

def generate_anomalous_traffic(device: str, n_samples: int = 20) -> pd.DataFrame:
    """Return a DataFrame of *anomalous* (attack-spike) rows for one device."""
    profile = DEVICE_PROFILES[device]
    rows = []
    for _ in range(n_samples):
        # Anomalies are 3-6× the normal mean
        multiplier = np.random.uniform(3, 6)
        rows.append({
            "device":        device,
            "packet_count":  profile["packet_count"][0] * multiplier,
            "request_freq":  profile["request_freq"][0]  * multiplier,
            "data_size_kb":  profile["data_size_kb"][0]  * multiplier,
            "label":         "anomaly",
        })
    return pd.DataFrame(rows)

def simulate_all_devices(output_path: str = "data/simulated_iot_data.csv") -> pd.DataFrame:
    """Simulate all devices and write combined CSV."""
    frames = []
    for device in DEVICE_PROFILES:
        frames.append(generate_normal_traffic(device))
        frames.append(generate_anomalous_traffic(device))

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)  # shuffle

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Simulator] Dataset saved → {output_path}  ({len(df)} rows)")
    return df

# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = simulate_all_devices()
    print(df.groupby(["device", "label"]).size().to_string())
