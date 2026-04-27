import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from complex_forward_model import TOPOGRAPHY_Z, X_COORDS, Z_MIN


CMAP_GEO = ListedColormap(["#4b0082", "#006400", "#e8c840", "#8B4513"])
GEO_LABELS = ["Host Rock", "Dense Body", "Basin", "Soil"]


def r2(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    denom = np.sum((obs - np.mean(obs)) ** 2) + 1e-12
    return float(1.0 - np.sum((pred - obs) ** 2) / denom)


def rmse(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def load_window(npz_path):
    data = np.load(npz_path)
    x = data["x_plot_km"]
    order = np.argsort(x)
    return {
        "path": npz_path,
        "x": x[order],
        "obs_g": data["obs_gravity"][order],
        "mod_g": data["model_gravity"][order],
        "obs_r": data["obs_radio"][:, order],
        "mod_r": data["model_radio"][:, order],
        "geology": data["geology"],
        "best": data["best_params"],
        "mc_std": data["mc_std"],
        "start": float(data["window_start_km"][0]),
        "end": float(data["window_end_km"][0]),
        "reverse": bool(int(data["reverse"][0])),
        "loss": float(data["best_loss"][0]),
    }


def _global_fault_km(win):
    local_km = float(win["best"][0]) / 1000.0
    if win["reverse"]:
        return win["end"] - local_km
    return win["start"] + local_km


def _geo_panel(ax, win):
    geo = np.clip(win["geology"] - 1, 0, 3)
    topo = TOPOGRAPHY_Z.copy()
    if win["reverse"]:
        geo = geo[:, ::-1]
        topo = topo[::-1]

    ax.imshow(
        geo,
        extent=[win["start"], win["end"], Z_MIN / 1000.0, 0.0],
        origin="lower",
        aspect="auto",
        cmap=CMAP_GEO,
        vmin=0,
        vmax=3,
        interpolation="none",
    )

    x_topo = win["start"] + (X_COORDS - X_COORDS.min()) / np.ptp(X_COORDS) * (win["end"] - win["start"])
    ax.plot(x_topo, topo / 1000.0, color="black", lw=1.2, alpha=0.75, label="Topo")
    ax.axvline(_global_fault_km(win), color="red", ls="--", lw=1.5, label="Fault")
    ax.set_xlim(win["start"], win["end"])
    ax.set_ylim(Z_MIN / 1000.0, 0.15)
    ax.set_ylabel("Elevation (km)")
    ax.set_title(
        f"Recovered Subsurface Model | Window {win['start']:.0f}-{win['end']:.0f} km"
        f" | reverse={win['reverse']} | fault={_global_fault_km(win):.2f} km",
        fontweight="bold",
    )
    handles = [Patch(color=CMAP_GEO(i / 3), label=GEO_LABELS[i]) for i in range(4)]
    handles += [
        plt.Line2D([0], [0], color="black", lw=1.2, label="Topo"),
        plt.Line2D([0], [0], color="red", ls="--", lw=1.2, label="Fault"),
    ]
    ax.legend(handles=handles, ncol=3, fontsize=8, loc="lower right")


def plot_single_window(win, out_png):
    x = win["x"]
    obs_g = win["obs_g"]
    mod_g = win["mod_g"]
    obs_r = win["obs_r"]
    mod_r = win["mod_r"]

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1.35, 1, 1, 1, 1, 0.55], hspace=0.52)
    fig.suptitle(
        f"Window {win['start']:.0f}-{win['end']:.0f} km Individual Inversion Profile",
        y=0.992,
        fontsize=15,
        fontweight="bold",
    )

    ax_geo = fig.add_subplot(gs[0])
    _geo_panel(ax_geo, win)

    ax_g = fig.add_subplot(gs[1])
    ax_g.plot(x, obs_g, color="black", lw=1.4, alpha=0.75, label="Observed gravity")
    ax_g.plot(x, mod_g, color="red", lw=1.9, label="Model gravity")
    ax_g.fill_between(x, mod_g - obs_g, 0, color="purple", alpha=0.16, label="Model - observed")
    ax_g.axhline(0, color="gray", ls=":", lw=0.8)
    ax_g.set_title(f"Gravity | R2={r2(obs_g, mod_g):.3f}, RMSE={rmse(obs_g, mod_g):.3f} mGal", fontweight="bold")
    ax_g.set_ylabel("mGal")
    ax_g.legend(fontsize=8)

    labels = [("K", "%"), ("U", "ppm"), ("Th", "ppm")]
    for idx, (name, unit) in enumerate(labels):
        ax = fig.add_subplot(gs[2 + idx])
        ax.plot(x, obs_r[idx], color="black", lw=1.3, alpha=0.75, label=f"Observed {name}")
        ax.plot(x, mod_r[idx], color="red", lw=1.8, label=f"Model {name}")
        ax.set_title(f"{name} | R2={r2(obs_r[idx], mod_r[idx]):.3f}, RMSE={rmse(obs_r[idx], mod_r[idx]):.3f}", fontweight="bold")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)

    ax_tbl = fig.add_subplot(gs[5])
    ax_tbl.axis("off")
    best = win["best"]
    std = win["mc_std"]
    rows = [
        ["Best loss", f"{win['loss']:.5f}"],
        ["Fault global", f"{_global_fault_km(win):.2f} km"],
        ["Fault local", f"{best[0] / 1000.0:.2f} +/- {std[0] / 1000.0:.2f} km"],
        ["rho_dense", f"{best[1]:.3f} +/- {std[1]:.3f} g/cc"],
        ["rho_basin", f"{best[2]:.3f} +/- {std[2]:.3f} g/cc"],
        ["Dense top", f"{best[3]:.0f} +/- {std[3]:.0f} m"],
        ["Radio peak 1", f"{best[4] / 1000.0:.2f} km"],
        ["Radio peak 2", f"{best[8] / 1000.0:.2f} km" if len(best) > 8 else "n/a"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Parameter", "Value"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)

    for ax in fig.axes:
        if ax is not ax_tbl:
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Distance along profile (km)")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_all_individual(results_dir):
    windows = [
        ("window_00_10", os.path.join(results_dir, "window_00_10", "real_data_best_results.npz")),
        ("window_10_20", os.path.join(results_dir, "window_10_20", "real_data_best_results.npz")),
        ("window_20_30", os.path.join(results_dir, "window_20_30", "real_data_best_results.npz")),
    ]
    outputs = []
    for name, path in windows:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        win = load_window(path)
        out_png = os.path.join(results_dir, f"{name}_individual_profile.png")
        plot_single_window(win, out_png)
        outputs.append(out_png)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Plot individual recovered profiles for all three windows.")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    )
    args = parser.parse_args()
    plot_all_individual(args.results_dir)


if __name__ == "__main__":
    main()
