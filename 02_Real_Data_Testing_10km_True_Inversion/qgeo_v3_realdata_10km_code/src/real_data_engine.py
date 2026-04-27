import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter

from complex_forward_model import run_complex_forward, X_MIN, X_MAX


PARAM_NAMES = [
    "fault_x_m",
    "rho_dense_gcc",
    "rho_basin_gcc",
    "dense_top_m",
    "radio_peak_x_m",
    "radio_peak_width_m",
    "radio_peak_gain",
    "radio_contact_width_m",
]

# Real-data inversion bounds. Densities are reported as physical g/cc.
PARAM_BOUNDS = np.array(
    [
        [1500.0, 8000.0],
        [0.05, 0.50],
        [0.02, 0.22],
        [-2600.0, -400.0],
        [400.0, 3500.0],
        [250.0, 2200.0],
        [0.0, 5.0],
        [80.0, 1400.0],
    ],
    dtype=float,
)


@dataclass
class RealDataConfig:
    csv_path: str = os.path.join("real_data", "real_testing_data.csv")
    results_dir: str = "results"
    window_start_km: str = "auto"
    window_length_km: float = 10.0
    auto_window_step_km: float = 0.5
    trend_degree: int = 2
    n_resample: int = 150
    radio_smooth_window: int = 9
    seed: int = 42
    maxiter: int = 45
    popsize: int = 9
    mc_samples: int = 16
    gravity_weight: float = 2.0
    radio_weight: float = 0.35
    allow_reverse: bool = True
    allow_negative_gravity_scale: bool = True
    allow_negative_radio_scale: bool = True
    use_realstyle_radio: bool = True


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _column(raw, aliases):
    names = [n.lower().strip() for n in raw.dtype.names]
    for alias in aliases:
        if alias in names:
            return raw[raw.dtype.names[names.index(alias)]]
    raise ValueError(f"Could not find any of {aliases}. CSV columns are {raw.dtype.names}")


def load_real_csv(csv_path):
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if raw.ndim == 0:
        raw = raw.reshape(1)

    x_km = _column(raw, ["dist_km", "distance_km", "x_km", "dist", "x"])
    g_mgal = _column(raw, ["g_mgal", "gravity", "bouguer", "bouguer_anomaly", "gz"])
    k_pct = _column(raw, ["k_pct", "k_percent", "k", "potassium"])
    u_ppm = _column(raw, ["u_ppm", "u", "uranium", "eu", "eu_ppm"])
    th_ppm = _column(raw, ["th_ppm", "th", "thorium", "eth", "eth_ppm"])

    valid = np.isfinite(x_km) & np.isfinite(g_mgal) & np.isfinite(k_pct)
    valid &= np.isfinite(u_ppm) & np.isfinite(th_ppm)
    order = np.argsort(x_km[valid])

    return {
        "x_km": x_km[valid][order],
        "g_mgal": g_mgal[valid][order],
        "k_pct": k_pct[valid][order],
        "u_ppm": u_ppm[valid][order],
        "th_ppm": th_ppm[valid][order],
    }


def robust_polynomial_trend(x_km, y, degree):
    x_norm = (x_km - np.mean(x_km)) / (np.ptp(x_km) + 1e-12)
    mask = np.isfinite(y)
    coeffs = None

    for _ in range(6):
        coeffs = np.polyfit(x_norm[mask], y[mask], degree)
        resid = y - np.polyval(coeffs, x_norm)
        mad = np.median(np.abs(resid[mask] - np.median(resid[mask]))) + 1e-12
        sigma = 1.4826 * mad
        new_mask = np.abs(resid) < 2.75 * sigma
        if new_mask.sum() < max(degree + 3, int(0.55 * len(y))):
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    trend = np.polyval(coeffs, x_norm)
    return trend, coeffs, mask


def _smooth(y, requested_window):
    if len(y) < 5:
        return y.copy()
    win = min(int(requested_window), len(y) - (1 - len(y) % 2))
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    if win >= len(y):
        win = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if win < 5:
        return y.copy()
    return savgol_filter(y, win, 2)


def build_window_dataset(data, config, window_start_km, reverse):
    x_km = data["x_km"]
    g_raw = data["g_mgal"]
    trend, coeffs, trend_mask = robust_polynomial_trend(
        x_km, g_raw, int(config.trend_degree)
    )
    g_resid = g_raw - trend

    start = float(window_start_km)
    end = start + float(config.window_length_km)
    mask = (x_km >= start) & (x_km <= end)
    if mask.sum() < 8:
        raise ValueError(f"Only {mask.sum()} points in window {start:g}-{end:g} km")

    x_win = x_km[mask]
    src = (x_win - x_win.min()) / (x_win.max() - x_win.min() + 1e-12) * 10000.0
    values = [
        g_resid[mask],
        data["k_pct"][mask],
        data["u_ppm"][mask],
        data["th_ppm"][mask],
    ]

    if reverse:
        src = 10000.0 - src
        order = np.argsort(src)
        src = src[order]
        values = [v[order] for v in values]

    x_model_m = np.linspace(X_MIN, X_MAX, int(config.n_resample))

    def resample(y):
        return interp1d(src, y, kind="linear", fill_value="extrapolate")(x_model_m)

    obs_g = resample(values[0])
    obs_r = np.vstack(
        [
            _smooth(resample(values[1]), config.radio_smooth_window),
            _smooth(resample(values[2]), config.radio_smooth_window),
            _smooth(resample(values[3]), config.radio_smooth_window),
        ]
    )

    if reverse:
        x_plot_km = end - (x_model_m / 10000.0) * (end - start)
    else:
        x_plot_km = start + (x_model_m / 10000.0) * (end - start)

    return {
        "x_model_m": x_model_m,
        "x_plot_km": x_plot_km,
        "obs_gravity": obs_g,
        "obs_radio": obs_r,
        "window_start_km": start,
        "window_end_km": end,
        "reverse": bool(reverse),
        "trend": trend,
        "trend_coeffs": coeffs,
        "trend_mask": trend_mask,
        "g_resid_full": g_resid,
    }


def calibrate_series(pred, obs, allow_negative_scale=True, nonnegative=False):
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    a = np.vstack([pred, np.ones_like(pred)]).T

    coef = np.linalg.lstsq(a, obs, rcond=None)[0]
    if not allow_negative_scale and coef[0] < 0:
        coef[0] = 0.0
        coef[1] = float(np.mean(obs))

    fit = a @ coef
    if nonnegative:
        fit = np.clip(fit, 0.0, None)

    rmse = float(np.sqrt(np.mean((fit - obs) ** 2)))
    denom = float(np.sum((obs - np.mean(obs)) ** 2)) + 1e-12
    r2 = float(1.0 - np.sum((fit - obs) ** 2) / denom)
    norm_mse = float(np.mean(((fit - obs) / (np.std(obs) + 1e-9)) ** 2))
    return fit, coef, {"rmse": rmse, "r2": r2, "norm_mse": norm_mse}


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def realstyle_radiometric_response(x_model_m, params):
    """Real-data radiometric proxy: contact contrast plus shallow enrichment.

    The original synthetic model used a single left/right step in K/U/Th. The
    Karnataka test profile has a localized radiometric high near the first
    contact, so this proxy keeps the geological contact but adds one shallow
    enrichment lobe. Per-channel scale/offset calibration maps the shared shape
    to K, U, and Th.
    """
    fault_x_m = float(params[0])
    peak_x_m = float(params[4])
    peak_width_m = max(float(params[5]), 50.0)
    peak_gain = max(float(params[6]), 0.0)
    contact_width_m = max(float(params[7]), 30.0)

    left_block = 1.0 - _sigmoid((x_model_m - fault_x_m) / contact_width_m)
    enrichment = np.exp(-0.5 * ((x_model_m - peak_x_m) / peak_width_m) ** 2)
    shoulder = np.exp(-0.5 * ((x_model_m - (fault_x_m + 1700.0)) / 800.0) ** 2)

    shape = 0.65 * left_block + peak_gain * enrichment + 0.10 * shoulder
    shape = (shape - np.min(shape)) / (np.ptp(shape) + 1e-12)
    return np.vstack([shape, shape, shape])


def params_to_forward(params):
    fault_x_m, rho_dense_gcc, rho_basin_gcc, dense_top_m = np.asarray(params, dtype=float)[:4]
    return {
        "fault_x_loc": fault_x_m,
        "rho_dense_scaled": rho_dense_gcc / 1000.0,
        "rho_basin_scaled": rho_basin_gcc / 1000.0,
        "dense_depth_top": dense_top_m,
        "noise_level": 0.0,
        "seed": 0,
    }


def evaluate_params(params, dataset, config):
    _, g_raw, k_raw, u_raw, th_raw, geology = run_complex_forward(
        **params_to_forward(params)
    )
    if config.use_realstyle_radio and len(params) >= 8:
        raw_radio = realstyle_radiometric_response(dataset["x_model_m"], params)
    else:
        raw_radio = np.vstack([k_raw, u_raw, th_raw])

    g_fit, g_coef, g_metrics = calibrate_series(
        g_raw,
        dataset["obs_gravity"],
        allow_negative_scale=config.allow_negative_gravity_scale,
        nonnegative=False,
    )

    radio_fit = []
    radio_coef = []
    radio_metrics = []
    for i in range(3):
        fit, coef, metrics = calibrate_series(
            raw_radio[i],
            dataset["obs_radio"][i],
            allow_negative_scale=config.allow_negative_radio_scale,
            nonnegative=True,
        )
        radio_fit.append(fit)
        radio_coef.append(coef)
        radio_metrics.append(metrics)

    radio_norm_mse = float(np.mean([m["norm_mse"] for m in radio_metrics]))
    loss = config.gravity_weight * g_metrics["norm_mse"] + config.radio_weight * radio_norm_mse

    return {
        "loss": float(loss),
        "raw_gravity": g_raw,
        "raw_radio": raw_radio,
        "model_gravity": g_fit,
        "model_radio": np.asarray(radio_fit),
        "gravity_coef": np.asarray(g_coef),
        "radio_coef": np.asarray(radio_coef),
        "gravity_metrics": g_metrics,
        "radio_metrics": radio_metrics,
        "geology": geology,
    }


def score_window(data, config, start_km, reverse):
    dataset = build_window_dataset(data, config, start_km, reverse)
    default_params = np.array([4500.0, 0.20, 0.15, -1200.0, 1700.0, 800.0, 2.5, 450.0])
    ev = evaluate_params(default_params, dataset, config)
    return ev["loss"], dataset, ev


def choose_window(data, config):
    x_min = float(np.min(data["x_km"]))
    x_max = float(np.max(data["x_km"]))
    last_start = x_max - float(config.window_length_km)
    starts = np.arange(x_min, last_start + 1e-9, float(config.auto_window_step_km))
    reverse_options = [False, True] if config.allow_reverse else [False]

    rows = []
    best = None
    for start in starts:
        for reverse in reverse_options:
            try:
                loss, dataset, ev = score_window(data, config, float(start), reverse)
            except Exception:
                continue
            row = [
                float(loss),
                float(start),
                float(start + config.window_length_km),
                int(reverse),
                float(ev["gravity_metrics"]["r2"]),
                float(np.mean([m["r2"] for m in ev["radio_metrics"]])),
            ]
            rows.append(row)
            if best is None or loss < best[0]:
                best = (loss, dataset, ev)

    if best is None:
        raise RuntimeError("Auto-window search failed. Try a fixed --window-start value.")
    return best[1], np.asarray(rows, dtype=float)


def make_objective(dataset, config):
    bounds = PARAM_BOUNDS.copy()

    def objective(params):
        params = np.asarray(params, dtype=float)
        if np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1]):
            return 1e9
        try:
            return evaluate_params(params, dataset, config)["loss"]
        except Exception:
            return 1e9

    return objective


def run_real_data_inversion(config):
    os.makedirs(config.results_dir, exist_ok=True)
    t0 = time.time()
    csv_path = os.path.abspath(config.csv_path)
    data = load_real_csv(csv_path)

    print("=" * 72)
    print("  QUANTUM GEOPHYSICAL INVERSION v3 - REAL DATA BEST PACKAGE")
    print("=" * 72)
    print(f"  CSV             : {csv_path}")
    print(f"  Rows            : {len(data['x_km'])}")
    print(f"  Profile         : {data['x_km'][0]:.2f} to {data['x_km'][-1]:.2f} km")
    print(f"  Trend degree    : {config.trend_degree}")

    if str(config.window_start_km).lower() == "auto":
        print("  Window          : auto search")
        dataset, scan_table = choose_window(data, config)
    else:
        start = float(config.window_start_km)
        dataset = build_window_dataset(data, config, start, reverse=False)
        scan_table = np.empty((0, 6), dtype=float)

    print(
        "  Selected window : "
        f"{dataset['window_start_km']:.2f}-{dataset['window_end_km']:.2f} km"
        f" | reverse={dataset['reverse']}"
    )
    print(
        "  Obs residual G  : "
        f"{dataset['obs_gravity'].min():.3f} to {dataset['obs_gravity'].max():.3f} mGal"
    )

    objective = make_objective(dataset, config)
    bounds = [tuple(row) for row in PARAM_BOUNDS]
    history = []

    def callback(xk, convergence):
        val = float(objective(xk))
        history.append([len(history), val, *map(float, xk), float(convergence)])
        if len(history) % 5 == 0:
            print(
                f"  iter {len(history):03d} | loss={val:.5f} | "
                f"fault={xk[0]:.0f} m | dense={xk[1]:.3f} g/cc | "
                f"basin={xk[2]:.3f} g/cc | top={xk[3]:.0f} m",
                flush=True,
            )
        return False

    print("\n[1/3] Global optimizer")
    de = differential_evolution(
        objective,
        bounds,
        seed=int(config.seed),
        maxiter=int(config.maxiter),
        popsize=int(config.popsize),
        tol=0.003,
        polish=False,
        updating="immediate",
        workers=1,
        callback=callback,
    )
    print(f"  DE best loss    : {de.fun:.6f}")

    print("\n[2/3] Local polish")
    local = minimize(
        objective,
        de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-8},
    )
    best_params = local.x if local.fun <= de.fun else de.x
    best_loss = float(min(local.fun, de.fun))
    best_eval = evaluate_params(best_params, dataset, config)
    print(f"  Final loss      : {best_loss:.6f}")

    print("\n[3/3] Monte Carlo uncertainty")
    mc_params = []
    rng = np.random.default_rng(int(config.seed) + 1000)
    g_sigma = max(best_eval["gravity_metrics"]["rmse"], 0.03 * np.std(dataset["obs_gravity"]))
    r_sigma = np.array(
        [max(m["rmse"], 0.03 * np.std(dataset["obs_radio"][i])) for i, m in enumerate(best_eval["radio_metrics"])]
    )

    for i in range(int(config.mc_samples)):
        mc_dataset = dict(dataset)
        mc_dataset["obs_gravity"] = dataset["obs_gravity"] + rng.normal(
            0.0, g_sigma, dataset["obs_gravity"].shape
        )
        mc_dataset["obs_radio"] = dataset["obs_radio"] + rng.normal(
            0.0, r_sigma[:, None], dataset["obs_radio"].shape
        )
        mc_obj = make_objective(mc_dataset, config)
        perturb = np.array([180.0, 0.025, 0.025, 80.0, 160.0, 90.0, 0.20, 60.0])
        start = best_params + rng.normal(0.0, perturb, size=len(best_params))
        start = np.clip(start, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        mc = minimize(
            mc_obj,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 70, "ftol": 1e-8},
        )
        mc_params.append(mc.x if mc.success else start)
        if (i + 1) % 4 == 0 or i == int(config.mc_samples) - 1:
            print(f"  MC {i + 1}/{config.mc_samples} done", flush=True)

    mc_params = np.asarray(mc_params, dtype=float)
    if len(mc_params):
        mc_mean = mc_params.mean(axis=0)
        mc_std = mc_params.std(axis=0)
    else:
        mc_mean = best_params.copy()
        mc_std = np.zeros_like(best_params)

    result_npz = os.path.join(config.results_dir, "real_data_best_results.npz")
    np.savez(
        result_npz,
        best_params=best_params,
        best_loss=np.array([best_loss]),
        mc_params=mc_params,
        mc_mean=mc_mean,
        mc_std=mc_std,
        x_model_m=dataset["x_model_m"],
        x_plot_km=dataset["x_plot_km"],
        obs_gravity=dataset["obs_gravity"],
        obs_radio=dataset["obs_radio"],
        model_gravity=best_eval["model_gravity"],
        model_radio=best_eval["model_radio"],
        raw_model_gravity=best_eval["raw_gravity"],
        raw_model_radio=best_eval["raw_radio"],
        gravity_coef=best_eval["gravity_coef"],
        radio_coef=best_eval["radio_coef"],
        realstyle_radio_raw=realstyle_radiometric_response(dataset["x_model_m"], best_params),
        geology=best_eval["geology"],
        scan_table=scan_table,
        history=np.asarray(history, dtype=float),
        full_x_km=data["x_km"],
        full_g_mgal=data["g_mgal"],
        full_k_pct=data["k_pct"],
        full_u_ppm=data["u_ppm"],
        full_th_ppm=data["th_ppm"],
        full_g_trend=dataset["trend"],
        full_g_resid=dataset["g_resid_full"],
        window_start_km=np.array([dataset["window_start_km"]]),
        window_end_km=np.array([dataset["window_end_km"]]),
        reverse=np.array([int(dataset["reverse"])]),
        trend_degree=np.array([config.trend_degree]),
    )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": time.time() - t0,
        "config": asdict(config),
        "selected_window_km": [dataset["window_start_km"], dataset["window_end_km"]],
        "reverse_profile_for_model": dataset["reverse"],
        "best_loss": best_loss,
        "best_parameters": dict(zip(PARAM_NAMES, best_params)),
        "mc_mean": dict(zip(PARAM_NAMES, mc_mean)),
        "mc_std": dict(zip(PARAM_NAMES, mc_std)),
        "gravity_calibration": {
            "scale": best_eval["gravity_coef"][0],
            "offset": best_eval["gravity_coef"][1],
            **best_eval["gravity_metrics"],
        },
        "radiometric_calibration": {
            name: {
                "scale": best_eval["radio_coef"][i][0],
                "offset": best_eval["radio_coef"][i][1],
                **best_eval["radio_metrics"][i],
            }
            for i, name in enumerate(["K", "U", "Th"])
        },
    }
    result_json = os.path.join(config.results_dir, "real_data_best_summary.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    print("\n" + "=" * 72)
    print("  REAL DATA RESULT SUMMARY")
    print("=" * 72)
    for name, val, sd in zip(PARAM_NAMES, best_params, mc_std):
        print(f"  {name:<16} {val:>12.4f} +/- {sd:.4f}")
    print(
        "  gravity fit     "
        f"R2={best_eval['gravity_metrics']['r2']:.3f} "
        f"RMSE={best_eval['gravity_metrics']['rmse']:.3f} mGal "
        f"scale={best_eval['gravity_coef'][0]:.5f}"
    )
    for i, name in enumerate(["K", "U", "Th"]):
        print(
            f"  {name:<16} R2={best_eval['radio_metrics'][i]['r2']:.3f} "
            f"RMSE={best_eval['radio_metrics'][i]['rmse']:.3f} "
            f"scale={best_eval['radio_coef'][i][0]:.5f}"
        )
    print(f"  Saved           {result_npz}")
    print(f"  Saved           {result_json}")
    print("=" * 72)

    return summary
