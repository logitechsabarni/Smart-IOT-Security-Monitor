"""
dashboard.py
Matplotlib-based dashboard for the IoT Security Monitor.

Shows four panels:
  1. Traffic over time (packet_count) per device, colour-coded by anomaly
  2. Bar chart – total vs alert count per device
  3. Scatter – request_freq vs data_size_kb, coloured by status
  4. Anomaly-score timeline (lower = more suspicious)

Run this AFTER simulating data:
    python simulation/device_simulator.py   # create CSV
    python ml_model/anomaly_detector.py     # train model
    python visualization/dashboard.py       # show dashboard
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from ml_model.anomaly_detector import predict, FEATURES

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#30363d",
})

PALETTE = {
    "Smart Camera":  "#58a6ff",
    "Smart Bulb":    "#3fb950",
    "Smart Sensor":  "#d2a8ff",
}
C_NORMAL  = "#3fb950"
C_ANOMALY = "#f85149"


def load_and_score(path: str = "data/simulated_iot_data.csv") -> pd.DataFrame:
    """Load CSV and run every row through the ML model."""
    df = pd.read_csv(path)
    results = df.apply(
        lambda r: pd.Series(predict(r["packet_count"], r["request_freq"], r["data_size_kb"])),
        axis=1,
    )
    df = pd.concat([df.reset_index(drop=True), results], axis=1)
    df["row_index"] = range(len(df))
    return df


def build_dashboard(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    fig.suptitle("🛡  Smart IoT Security Monitor — Anomaly Dashboard",
                 fontsize=17, fontweight="bold", color="#e6edf3", y=1.01)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # ── Panel 1: packet_count timeline ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # full-width top row
    ax1.set_title("Packet Count Over Time — colour = anomaly status", fontsize=12, pad=8)
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Packet count")
    ax1.grid(True)

    for device, grp in df.groupby("device"):
        color = PALETTE[device]
        # Normal points
        norm = grp[~grp["anomaly"]]
        ax1.scatter(norm["row_index"], norm["packet_count"],
                    c=color, alpha=0.55, s=20, label=f"{device} – normal")
        # Anomaly points (red halo)
        anom = grp[grp["anomaly"]]
        ax1.scatter(anom["row_index"], anom["packet_count"],
                    c=C_ANOMALY, edgecolors="white", linewidths=0.6,
                    marker="X", s=70, zorder=5, label=f"{device} – ⚠ alert")

    ax1.legend(fontsize=8, ncol=3, loc="upper left")

    # ── Panel 2: bar — alerts per device ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Total vs Alert Count per Device", fontsize=12, pad=8)
    ax2.grid(True, axis="y")

    devices = list(PALETTE.keys())
    x       = np.arange(len(devices))
    width   = 0.35

    totals = [len(df[df["device"] == d]) for d in devices]
    alerts = [df[(df["device"] == d) & df["anomaly"]].shape[0] for d in devices]

    bars1 = ax2.bar(x - width/2, totals, width, color=[PALETTE[d] for d in devices],
                    alpha=0.7, label="Total samples")
    bars2 = ax2.bar(x + width/2, alerts, width, color=C_ANOMALY,
                    alpha=0.85, label="Alerts 🔴")

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace(" ", "\n") for d in devices], fontsize=9)
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)

    # Label bar values
    for bar in list(bars1) + list(bars2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    # ── Panel 3: scatter — request_freq vs data_size_kb ──────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Request Freq vs Data Size (Normal / Suspicious)", fontsize=12, pad=8)
    ax3.set_xlabel("Request frequency (req/s)")
    ax3.set_ylabel("Data size (KB)")
    ax3.grid(True)

    for device, grp in df.groupby("device"):
        color = PALETTE[device]
        norm  = grp[~grp["anomaly"]]
        anom  = grp[grp["anomaly"]]
        ax3.scatter(norm["request_freq"], norm["data_size_kb"],
                    c=color, alpha=0.50, s=22, label=device)
        ax3.scatter(anom["request_freq"], anom["data_size_kb"],
                    c=C_ANOMALY, marker="X", s=70, edgecolors="white",
                    linewidths=0.5, zorder=6)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[d], markersize=8, label=d)
        for d in PALETTE
    ] + [Line2D([0], [0], marker="X", color="w", markerfacecolor=C_ANOMALY,
                markersize=9, label="⚠ Anomaly")]
    ax3.legend(handles=legend_elements, fontsize=8)

    # ── Alert summary text box ────────────────────────────────────────────────
    total_alerts = df["anomaly"].sum()
    total_rows   = len(df)
    pct          = total_alerts / total_rows * 100
    summary = (
        f"Total samples : {total_rows}\n"
        f"Total alerts  : {total_alerts}\n"
        f"Alert rate    : {pct:.1f} %\n"
        "\nDevice breakdown:\n" +
        "\n".join(
            f"  {d}: {df[(df['device']==d) & df['anomaly']].shape[0]} alerts"
            for d in devices
        )
    )
    fig.text(0.5, -0.04, summary, ha="center", fontsize=9.5,
             color="#8b949e",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22",
                       edgecolor="#30363d", alpha=0.9))

    plt.savefig("visualization/iot_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("[Dashboard] Saved → visualization/iot_dashboard.png")
    plt.show()


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("visualization", exist_ok=True)
    print("[Dashboard] Loading and scoring dataset …")
    df = load_and_score()
    print(f"[Dashboard] {len(df)} rows | {df['anomaly'].sum()} anomalies detected")
    build_dashboard(df)
