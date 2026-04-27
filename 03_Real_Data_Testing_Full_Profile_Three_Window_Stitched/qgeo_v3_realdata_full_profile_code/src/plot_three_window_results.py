import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from complex_forward_model import Z_MIN


CMAP_GEO = ListedColormap(["#4b0082", "#006400", "#e8c840", "#8B4513"])
GEO_LABELS = ["Host Rock", "Dense Body", "Basin", "Soil"]


def _r2(obs, pred):
    denom = float(np.sum((obs - np.mean(obs)) ** 2)) + 1e-12
    return float(1.0 - np.sum((pred - obs) ** 2) / denom)


def _rmse(obs, pred):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def _load_window(path):
    data = np.load(path)
    x = data["x_plot_km"]
    order = np.argsort(x)
    return {
        "path": path,
        "x": x[order],
        "obs_g": data["obs_gravity"][order],
        "mod_g": data["model_gravity"][order],
        "obs_r": data["obs_radio"][:, order],
        "mod_r": data["model_radio"][:, order],
        "best": data["best_params"],
        "mc_std": data["mc_std"],
        "geology": data["geology"],
        "reverse": bool(int(data["reverse"][0])),
        "start": float(data["window_start_km"][0]),
        "end": float(data["window_end_km"][0]),
        "loss": float(data["best_loss"][0]),
        "full_x": data["full_x_km"],
        "full_g": data["full_g_mgal"],
        "full_trend": data["full_g_trend"],
        "full_resid": data["full_g_resid"],
    }


def _concat_windows(windows, key):
    xs = []
    ys = []
    last = -np.inf
    for win in windows:
        x = win["x"]
        y = win[key]
        keep = x > last + 1e-9
        xs.append(x[keep])
        ys.append(y[..., keep] if y.ndim == 2 else y[keep])
        last = max(last, float(np.max(x)))
    x_all = np.concatenate(xs)
    if ys[0].ndim == 2:
        return x_all, np.concatenate(ys, axis=1)
    return x_all, np.concatenate(ys)


def build_three_window_plot(results_dir):
    window_dirs = [
        os.path.join(results_dir, "window_00_10", "real_data_best_results.npz"),
        os.path.join(results_dir, "window_10_20", "real_data_best_results.npz"),
        os.path.join(results_dir, "window_20_30", "real_data_best_results.npz"),
    ]
    missing = [p for p in window_dirs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing window result files: " + ", ".join(missing))

    windows = [_load_window(p) for p in window_dirs]
    windows.sort(key=lambda w: w["start"])

    xg, obs_g = _concat_windows(windows, "obs_g")
    _, mod_g = _concat_windows(windows, "mod_g")
    xr, obs_r = _concat_windows(windows, "obs_r")
    _, mod_r = _concat_windows(windows, "mod_r")

    full = windows[0]
    fig = plt.figure(figsize=(18, 22))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig.suptitle(
        "Quantum Geophysical Inversion v3 - Three-Window Unified Real-Data Result\n"
        "Three independent 10 km inversions stitched across the 30 km profile",
        y=0.992,
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(6, 3, hspace=0.62, wspace=0.32)
    fig.subplots_adjust(top=0.94)

    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(full["full_x"], full["full_g"], color="#2878B5", lw=1.3, label="Raw Bouguer gravity")
    ax_full.plot(full["full_x"], full["full_trend"], "r--", lw=1.4, label="Regional trend")
    colors = ["#F59E0B", "#22C55E", "#60A5FA"]
    for i, win in enumerate(windows):
        ax_full.axvspan(win["start"], win["end"], color=colors[i], alpha=0.14, label=f"Window {i+1}: {win['start']:.0f}-{win['end']:.0f} km")
    ax_full.set_title("Full Profile With Three 10 km Inversion Windows", fontweight="bold")
    ax_full.set_xlabel("Distance along profile (km)")
    ax_full.set_ylabel("Bouguer anomaly (mGal)")
    ax_full.legend(fontsize=8, ncol=2)
    ax_resid = ax_full.twinx()
    ax_resid.plot(full["full_x"], full["full_resid"], color="green", alpha=0.55, lw=1.0, label="Residual")
    ax_resid.axhline(0, color="green", ls=":", lw=0.8)
    ax_resid.set_ylabel("Residual anomaly (mGal)", color="green")
    ax_resid.tick_params(axis="y", labelcolor="green")

    ax_geo = fig.add_subplot(gs[1, :])
    for win in windows:
        geo = np.clip(win["geology"] - 1, 0, 3)
        if win["reverse"]:
            geo = geo[:, ::-1]
        ax_geo.imshow(
            geo,
            extent=[win["start"], win["end"], Z_MIN / 1000.0, 0],
            origin="lower",
            aspect="auto",
            cmap=CMAP_GEO,
            vmin=0,
            vmax=3,
            interpolation="none",
            alpha=0.96,
        )
        fault_global = win["end"] - win["best"][0] / 1000.0 if win["reverse"] else win["start"] + win["best"][0] / 1000.0
        ax_geo.axvline(fault_global, color="red", ls="--", lw=1.2)
    ax_geo.set_title("Stitched Geological Sections (one local model per 10 km window)", fontweight="bold")
    ax_geo.set_xlabel("Distance along profile (km)")
    ax_geo.set_ylabel("Elevation (km)")
    ax_geo.set_xlim(0, 30)
    ax_geo.set_ylim(Z_MIN / 1000.0, 0.1)
    handles = [Patch(color=CMAP_GEO(i / 3), label=GEO_LABELS[i]) for i in range(4)]
    handles.append(plt.Line2D([0], [0], color="red", ls="--", lw=1.2, label="Fault"))
    ax_geo.legend(handles=handles, ncol=5, fontsize=8, loc="lower right")

    ax_g = fig.add_subplot(gs[2, :])
    ax_g.plot(xg, obs_g, color="black", lw=1.2, label="Observed residual anomaly")
    ax_g.plot(xg, mod_g, color="red", lw=1.8, label="Stitched model")
    for win in windows[1:]:
        ax_g.axvline(win["start"], color="gray", ls=":", lw=1.0)
    ax_g.fill_between(xg, mod_g - obs_g, 0, color="purple", alpha=0.16, label="Model - observed")
    ax_g.set_title(f"Unified Gravity Fit: R2={_r2(obs_g, mod_g):.3f}, RMSE={_rmse(obs_g, mod_g):.3f} mGal", fontweight="bold")
    ax_g.set_xlabel("Distance along profile (km)")
    ax_g.set_ylabel("Residual gravity anomaly (mGal)")
    ax_g.legend(fontsize=8)

    labels = [("K", "%"), ("U", "ppm"), ("Th", "ppm")]
    for i, (name, unit) in enumerate(labels):
        ax = fig.add_subplot(gs[3, i])
        ax.plot(xr, obs_r[i], color="black", lw=1.1, alpha=0.75, label=f"Observed {name}")
        ax.plot(xr, mod_r[i], color="red", lw=1.7, label=f"Stitched {name}")
        for win in windows[1:]:
            ax.axvline(win["start"], color="gray", ls=":", lw=0.9)
        ax.set_title(f"{name} Fit: R2={_r2(obs_r[i], mod_r[i]):.3f}", fontweight="bold")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(fontsize=8)

    ax_tbl = fig.add_subplot(gs[4:, :])
    ax_tbl.axis("off")
    rows = []
    for idx, win in enumerate(windows, start=1):
        fault_global = win["end"] - win["best"][0] / 1000.0 if win["reverse"] else win["start"] + win["best"][0] / 1000.0
        rows.append(
            [
                f"W{idx}: {win['start']:.0f}-{win['end']:.0f} km",
                f"{win['loss']:.4f}",
                str(win["reverse"]),
                f"{fault_global:.2f} km",
                f"{win['best'][1]:.3f}",
                f"{win['best'][2]:.3f}",
                f"{win['best'][3]:.0f}",
                f"{win['best'][4] / 1000.0:.2f} km",
            ]
        )
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Window", "Loss", "Reversed", "Fault global", "rho_dense", "rho_basin", "dense top", "radio peak"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    summary = {
        "windows": rows,
        "unified_metrics": {
            "gravity_r2": _r2(obs_g, mod_g),
            "gravity_rmse_mgal": _rmse(obs_g, mod_g),
            "k_r2": _r2(obs_r[0], mod_r[0]),
            "u_r2": _r2(obs_r[1], mod_r[1]),
            "th_r2": _r2(obs_r[2], mod_r[2]),
        },
        "limitations": [
            "Each 10 km section has independent calibration and local model coordinates.",
            "The stitched section is a unified interpretation, not a single globally coupled inversion.",
            "Sharp discontinuities at 10 km and 20 km indicate boundary/calibration mismatch, not necessarily geology.",
        ],
    }

    out_png = os.path.join(results_dir, "unified_three_window_plot.png")
    out_json = os.path.join(results_dir, "three_window_summary.json")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_png}")
    print(f"Saved {out_json}")
    return out_png, out_json


def main():
    parser = argparse.ArgumentParser(description="Plot stitched three-window inversion results.")
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = parser.parse_args()
    build_three_window_plot(args.results_dir)


if __name__ == "__main__":
    main()
