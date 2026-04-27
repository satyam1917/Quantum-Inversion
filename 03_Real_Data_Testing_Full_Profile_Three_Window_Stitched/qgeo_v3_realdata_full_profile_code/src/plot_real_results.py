import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from complex_forward_model import TOPOGRAPHY_Z, X_COORDS, X_MAX, X_MIN, Z_MIN


CMAP_GEO = ListedColormap(["#4b0082", "#006400", "#e8c840", "#8B4513"])
GEO_LABELS = ["Host Rock", "Dense Body", "Basin", "Soil"]


def _r2(obs, pred):
    denom = float(np.sum((obs - np.mean(obs)) ** 2)) + 1e-12
    return float(1.0 - np.sum((pred - obs) ** 2) / denom)


def _rmse(obs, pred):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def _geo_panel(ax, geology, best_params, title):
    geo_plot = np.clip(geology - 1, 0, 3)
    ax.imshow(
        geo_plot,
        extent=[X_MIN / 1000.0, X_MAX / 1000.0, Z_MIN / 1000.0, 0.0],
        origin="lower",
        aspect="auto",
        cmap=CMAP_GEO,
        interpolation="none",
        vmin=0,
        vmax=3,
    )
    ax.plot(X_COORDS / 1000.0, TOPOGRAPHY_Z / 1000.0, "k-", lw=1.2, alpha=0.75)
    ax.axvline(best_params[0] / 1000.0, color="red", ls="--", lw=1.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Model distance (km)")
    ax.set_ylabel("Elevation (km)")
    ax.set_ylim(Z_MIN / 1000.0, 0.15)
    handles = [Patch(color=CMAP_GEO(i / 3), label=GEO_LABELS[i]) for i in range(4)]
    handles.append(plt.Line2D([0], [0], color="k", lw=1.2, label="Topo"))
    handles.append(plt.Line2D([0], [0], color="red", ls="--", lw=1.2, label="Fault"))
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")


def plot_results(results_dir):
    npz_path = os.path.join(results_dir, "real_data_best_results.npz")
    json_path = os.path.join(results_dir, "real_data_best_summary.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}. Run src/run_real_data.py first.")

    data = np.load(npz_path)
    summary = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    best = data["best_params"]
    mc_std = data["mc_std"]
    x_plot = data["x_plot_km"]
    order = np.argsort(x_plot)

    obs_g = data["obs_gravity"][order]
    pred_g = data["model_gravity"][order]
    obs_r = data["obs_radio"][:, order]
    pred_r = data["model_radio"][:, order]
    x_fit = x_plot[order]

    win_start = float(data["window_start_km"][0])
    win_end = float(data["window_end_km"][0])
    reverse = bool(int(data["reverse"][0]))
    trend_degree = int(data["trend_degree"][0])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        "Quantum Geophysical Inversion v3 - Real Data Calibrated v4 Results\n"
        f"Window {win_start:.1f}-{win_end:.1f} km | reverse={reverse} | "
        f"fault={best[0] / 1000.0:.2f}+/-{mc_std[0] / 1000.0:.2f} km model | "
        f"dense={best[1]:.3f}+/-{mc_std[1]:.3f} g/cc | "
        f"basin={best[2]:.3f}+/-{mc_std[2]:.3f} g/cc | "
        f"top={best[3]:.0f}+/-{mc_std[3]:.0f} m",
        y=0.992,
        fontsize=12,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(5, 2, hspace=0.55, wspace=0.32)
    fig.subplots_adjust(top=0.93)

    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(data["full_x_km"], data["full_g_mgal"], color="#2878B5", lw=1.4, label="Raw Bouguer gravity")
    ax_full.plot(data["full_x_km"], data["full_g_trend"], "r--", lw=1.5, label=f"Robust regional trend (degree {trend_degree})")
    ax_full.axvspan(win_start, win_end, color="orange", alpha=0.16, label="Selected 10 km window")
    ax_full.set_title("Full Profile: Bouguer Gravity and Removed Regional Trend", fontweight="bold")
    ax_full.set_xlabel("Distance along profile (km)")
    ax_full.set_ylabel("Bouguer anomaly (mGal)")
    ax_full.legend(loc="upper left", fontsize=9)

    ax_resid = ax_full.twinx()
    ax_resid.plot(data["full_x_km"], data["full_g_resid"], color="green", alpha=0.55, lw=1.0, label="Residual")
    ax_resid.axhline(0, color="green", ls=":", lw=0.8, alpha=0.5)
    ax_resid.set_ylabel("Residual anomaly (mGal)", color="green")
    ax_resid.tick_params(axis="y", labelcolor="green")
    ax_resid.legend(loc="upper right", fontsize=8)

    ax_geo = fig.add_subplot(gs[1, 0])
    _geo_panel(
        ax_geo,
        data["geology"],
        best,
        "Recovered Geological Model\n(expanded-depth real-data fit)",
    )

    ax_scan = fig.add_subplot(gs[1, 1])
    scan = data["scan_table"]
    if scan.size:
        normal = scan[scan[:, 3] == 0]
        flipped = scan[scan[:, 3] == 1]
        if len(normal):
            ax_scan.plot(normal[:, 1], normal[:, 0], "o-", ms=4, label="Normal profile")
        if len(flipped):
            ax_scan.plot(flipped[:, 1], flipped[:, 0], "s-", ms=4, label="Reversed profile")
        ax_scan.axvline(win_start, color="red", ls="--", lw=1.2, label="Selected")
        ax_scan.set_xlabel("Window start (km)")
        ax_scan.set_ylabel("Template score (lower is better)")
        ax_scan.set_title("Auto-Window Screening", fontweight="bold")
        ax_scan.legend(fontsize=8)
    else:
        ax_scan.axis("off")
        ax_scan.text(0.5, 0.5, "Fixed window run", ha="center", va="center", fontsize=13)

    ax_g = fig.add_subplot(gs[2, :])
    ax_g.plot(x_fit, obs_g, color="black", lw=1.3, alpha=0.72, label="Observed residual anomaly")
    ax_g.plot(x_fit, pred_g, color="red", lw=2.0, label="Recovered model after survey calibration")
    ax_g.fill_between(x_fit, pred_g - obs_g, 0, color="purple", alpha=0.18, label="Model - observed")
    ax_g.axhline(0, color="gray", ls=":", lw=0.9)
    ax_g.set_title(
        f"Gravity Fit: R2={_r2(obs_g, pred_g):.3f}, RMSE={_rmse(obs_g, pred_g):.3f} mGal",
        fontweight="bold",
    )
    ax_g.set_xlabel("Distance along profile (km)")
    ax_g.set_ylabel("Residual gravity anomaly (mGal)")
    ax_g.legend(fontsize=9)

    ax_k = fig.add_subplot(gs[3, 0])
    ax_u = fig.add_subplot(gs[3, 1])
    channels = [("K", "%", ax_k), ("U", "ppm", ax_u)]
    for idx, (name, unit, ax) in enumerate(channels):
        ax.plot(x_fit, obs_r[idx], color="black", lw=1.2, alpha=0.72, label=f"Observed {name} (smoothed)")
        ax.plot(x_fit, pred_r[idx], color="red", lw=1.8, label=f"Recovered {name}")
        ax.set_title(
            f"Radiometric Fit: {name} | R2={_r2(obs_r[idx], pred_r[idx]):.3f}",
            fontweight="bold",
        )
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(fontsize=8)

    ax_th = fig.add_subplot(gs[4, 0])
    ax_tbl = fig.add_subplot(gs[4, 1])
    ax_th.plot(x_fit, obs_r[2], color="black", lw=1.2, alpha=0.72, label="Observed Th (smoothed)")
    ax_th.plot(x_fit, pred_r[2], color="red", lw=1.8, label="Recovered Th")
    ax_th.set_title(f"Radiometric Fit: Th | R2={_r2(obs_r[2], pred_r[2]):.3f}", fontweight="bold")
    ax_th.set_xlabel("Distance (km)")
    ax_th.set_ylabel("Th (ppm)")
    ax_th.legend(fontsize=8)

    ax_tbl.axis("off")
    gcal = summary.get("gravity_calibration", {})
    rows = [
        ["Window", f"{win_start:.1f}-{win_end:.1f} km"],
        ["Reverse profile", str(reverse)],
        ["Fault x (model)", f"{best[0]:.0f} +/- {mc_std[0]:.0f} m"],
        ["Fault x (window)", f"{best[0] / 1000.0:.2f} +/- {mc_std[0] / 1000.0:.2f} km"],
        ["rho_dense", f"{best[1]:.4f} +/- {mc_std[1]:.4f} g/cc"],
        ["rho_basin", f"{best[2]:.4f} +/- {mc_std[2]:.4f} g/cc"],
        ["Dense top", f"{best[3]:.0f} +/- {mc_std[3]:.0f} m"],
        ["Radio peak x", f"{best[4] / 1000.0:.2f} +/- {mc_std[4] / 1000.0:.2f} km"],
        ["Radio peak width", f"{best[5]:.0f} +/- {mc_std[5]:.0f} m"],
        ["Best loss", f"{float(data['best_loss'][0]):.5f}"],
        ["Gravity scale", f"{float(gcal.get('scale', data['gravity_coef'][0])):.5f}"],
    ]
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.25, 1.75)

    out_png = os.path.join(results_dir, "real_data_best_plot.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_png}")
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Plot calibrated real-data inversion.")
    parser.add_argument(
        "--results-dir",
        default=os.path.abspath(os.path.join(SRC_DIR, "..", "results")),
    )
    args = parser.parse_args()
    plot_results(args.results_dir)


if __name__ == "__main__":
    main()
