import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunConfig:
    n_points: int = 100
    train_ratio: float = 0.8
    seed: int = 42
    alpha: float = 1e-6
    data_path: str = "data/linear_regression_data.csv"
    results_dir: str = "results"


def ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta):
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def apply_one_qubit_gate(state, gate, qubit, n_qubits):
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def apply_cnot(state, control, target, n_qubits):
    out = state.copy()
    for index, amplitude in enumerate(state):
        if ((index >> (n_qubits - 1 - control)) & 1) == 1:
            flipped = index ^ (1 << (n_qubits - 1 - target))
            out[flipped] = state[index]
            out[index] = state[flipped]
    return out


def expectation_z(state, qubit, n_qubits):
    exp_val = 0.0
    for index, amplitude in enumerate(state):
        bit = (index >> (n_qubits - 1 - qubit)) & 1
        sign = 1.0 if bit == 0 else -1.0
        exp_val += sign * float(np.abs(amplitude) ** 2)
    return exp_val


def expectation_zz(state, q0, q1, n_qubits):
    exp_val = 0.0
    for index, amplitude in enumerate(state):
        b0 = (index >> (n_qubits - 1 - q0)) & 1
        b1 = (index >> (n_qubits - 1 - q1)) & 1
        sign = 1.0 if b0 == b1 else -1.0
        exp_val += sign * float(np.abs(amplitude) ** 2)
    return exp_val


def quantum_features(x_scaled):
    n_qubits = 2
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0

    angles = [np.pi * x_scaled, 0.5 * np.pi * x_scaled]
    for q, angle in enumerate(angles):
        state = apply_one_qubit_gate(state, ry(angle), q, n_qubits)
        state = apply_one_qubit_gate(state, rz(0.25 * angle), q, n_qubits)
    state = apply_cnot(state, 0, 1, n_qubits)
    state = apply_one_qubit_gate(state, ry(0.35 * np.pi * x_scaled), 1, n_qubits)

    z0 = expectation_z(state, 0, n_qubits)
    z1 = expectation_z(state, 1, n_qubits)
    zz = expectation_zz(state, 0, 1, n_qubits)
    return np.array([1.0, x_scaled, z0, z1, zz], dtype=float)


def make_dataset(cfg):
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(-1.0, 1.0, cfg.n_points)
    y_clean = 1.85 * x - 0.35
    y = y_clean + rng.normal(0.0, 0.075, size=cfg.n_points)

    order = rng.permutation(cfg.n_points)
    n_train = int(round(cfg.train_ratio * cfg.n_points))
    split = np.array(["test"] * cfg.n_points, dtype=object)
    split[order[:n_train]] = "train"
    return x, y, y_clean, split


def save_dataset(path, x, y, y_clean, split):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "y_clean", "split"])
        for row in zip(x, y, y_clean, split):
            writer.writerow(row)


def load_dataset(path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return rows["x"], rows["y"], rows["y_clean"], rows["split"]


def fit_quantum_regressor(x_train, y_train, alpha):
    phi = np.vstack([quantum_features(xi) for xi in x_train])
    regularizer = alpha * np.eye(phi.shape[1])
    weights = np.linalg.solve(phi.T @ phi + regularizer, phi.T @ y_train)
    return weights


def predict(x, weights):
    phi = np.vstack([quantum_features(xi) for xi in x])
    return phi @ weights


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_results(path, x, y, y_clean, split, weights, train_metrics, test_metrics):
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    pred_grid = predict(grid, weights)

    train = split == "train"
    test = split == "test"

    plt.figure(figsize=(10, 6), dpi=160)
    plt.scatter(x[train], y[train], s=34, color="#1f77b4", label="Training data")
    plt.scatter(x[test], y[test], s=42, color="#ff7f0e", marker="s", label="Testing data")
    plt.plot(grid, 1.85 * grid - 0.35, color="black", lw=2, ls="--", label="True linear trend")
    plt.plot(grid, pred_grid, color="#d62728", lw=2.5, label="Quantum regression fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        "Quantum Linear Regression | "
        f"Train R2={train_metrics['r2']:.3f}, Test R2={test_metrics['r2']:.3f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum feature-map linear regression")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--data-path", default="data/linear_regression_data.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig(
        n_points=args.n_points,
        train_ratio=args.train_ratio,
        seed=args.seed,
        alpha=args.alpha,
        data_path=args.data_path,
        results_dir=args.results_dir,
    )

    if args.regenerate or not os.path.exists(cfg.data_path):
        x, y, y_clean, split = make_dataset(cfg)
        save_dataset(cfg.data_path, x, y, y_clean, split)
    else:
        x, y, y_clean, split = load_dataset(cfg.data_path)

    train = split == "train"
    test = split == "test"
    weights = fit_quantum_regressor(x[train], y[train], cfg.alpha)
    pred_train = predict(x[train], weights)
    pred_test = predict(x[test], weights)

    train_metrics = metrics(y[train], pred_train)
    test_metrics = metrics(y[test], pred_test)

    os.makedirs(cfg.results_dir, exist_ok=True)
    plot_path = os.path.join(cfg.results_dir, "quantum_linear_regression.png")
    plot_results(plot_path, x, y, y_clean, split, weights, train_metrics, test_metrics)

    np.savez(
        os.path.join(cfg.results_dir, "quantum_linear_model.npz"),
        weights=weights,
        x=x,
        y=y,
        y_clean=y_clean,
        split=split,
    )
    summary = {
        "model": "quantum linear regression",
        "config": asdict(cfg),
        "n_train": int(np.sum(train)),
        "n_test": int(np.sum(test)),
        "weights": weights.tolist(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "dataset": cfg.data_path,
            "plot": plot_path,
            "model": os.path.join(cfg.results_dir, "quantum_linear_model.npz"),
        },
    }
    with open(os.path.join(cfg.results_dir, "quantum_linear_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Quantum linear regression complete")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
