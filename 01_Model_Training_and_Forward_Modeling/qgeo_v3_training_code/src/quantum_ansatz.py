# =============================================================================
# quantum_ansatz.py  (v3 CORRECTED)
# PARAM_BOUNDS updated to match SCALED density convention in forward model.
# =============================================================================

import numpy as np
import pennylane as qml

N_QUBITS = 8
N_LAYERS = 4

def _make_device():
    try:
        import pennylane_lightning  # noqa
        d = qml.device("lightning.qubit", wires=N_QUBITS)
        print("  [Quantum] Using lightning.qubit")
        return d
    except Exception:
        d = qml.device("default.qubit", wires=N_QUBITS)
        print("  [Quantum] Using default.qubit (fallback)")
        return d

_dev = _make_device()

# ---------------------------------------------------------------------------
# Parameter bounds — MUST match density scaling in complex_forward_model.py
# rho_dense and rho_basin are in SCALED units (g/cc × 1e-3)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = [
    (3000.0,   7000.0),     # p0: fault_x_loc (m)
    (0.00015,  0.00025),    # p1: rho_dense_scaled (g/cc × 1e-3) → 0.15-0.25 g/cc
    (0.00010,  0.00020),    # p2: rho_basin_scaled (g/cc × 1e-3) → 0.10-0.20 g/cc
    (-1500.0,  -500.0),     # p3: dense_depth_top (m)
]

PARAM_DEFAULTS = np.array([4500.0, 0.00020, 0.00015, -1000.0])
PARAM_NAMES    = ['fault_x_loc (m)', 'rho_dense (scaled)', 'rho_basin (scaled)', 'dense_top (m)']

# Physical equivalents for reporting
def scaled_to_physical(scaled_params):
    """Convert scaled density params to physical g/cc for human-readable output."""
    p = scaled_params.copy()
    p[1] = p[1] * 1000.0  # → g/cc
    p[2] = p[2] * 1000.0  # → g/cc
    return p


def denormalize(norm):
    out = np.empty(4)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        out[i] = float(np.clip(norm[i], 0., 1.)) * (hi - lo) + lo
    return out


def normalize(phys):
    out = np.empty(4)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        out[i] = (phys[i] - lo) / (hi - lo)
    return np.clip(out, 0., 1.)


@qml.qnode(_dev, interface="numpy")
def _circuit(weights):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def get_weight_shape():
    return qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS)


def circuit_to_physical(weights):
    expvals = _circuit(weights)
    norm    = np.array([(float(e) + 1.) / 2. for e in expvals])
    return denormalize(norm)


def weights_from_physical(phys, noise_std=0.3):
    norm        = normalize(phys)
    target      = norm * 2. - 1.
    shape       = get_weight_shape()
    base_angles = np.arccos(np.clip(target, -0.99, 0.99))
    weights     = np.zeros(shape)
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            weights[layer, qubit, 1] = base_angles[qubit % 4]
        weights[layer] += np.random.normal(0., noise_std, weights[layer].shape)
    return weights


def random_weights(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-np.pi, np.pi, get_weight_shape())


def lhs_weights(n_samples, seed=0):
    """Latin Hypercube Sampling for diverse multi-start initialisation."""
    rng   = np.random.default_rng(seed)
    shape = get_weight_shape()
    n_w   = int(np.prod(shape))
    bins  = np.arange(n_samples)
    result = []
    for _ in range(n_samples):
        perm   = np.array([rng.permutation(bins) for _ in range(n_w)], dtype=float).T
        w_flat = (perm[0] + rng.uniform(0, 1, n_w)) / n_samples
        w_flat = w_flat * 2 * np.pi - np.pi
        result.append(w_flat.reshape(shape))
    return result


if __name__ == "__main__":
    print("Testing quantum ansatz...")
    w = random_weights(seed=0)
    p = circuit_to_physical(w)
    phys = scaled_to_physical(p)
    print(f"  Weight shape : {w.shape}  ({w.size} params)")
    print(f"  Decoded (scaled) : fault_x={p[0]:.1f}  rho_d={p[1]:.5f}  rho_b={p[2]:.5f}  top={p[3]:.1f}")
    print(f"  Decoded (g/cc)   : rho_dense={phys[1]:.3f} g/cc  rho_basin={phys[2]:.3f} g/cc")
    lhs = lhs_weights(5, seed=0)
    print(f"  LHS samples  : {len(lhs)}")
    print("  PASS")
