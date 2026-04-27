# =============================================================================
# run_inversion.py  (v3 CORRECTED)
# =============================================================================

import numpy as np
import os, sys, time

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from complex_forward_model import run_complex_forward
from quantum_ansatz         import (circuit_to_physical, random_weights,
                                     N_QUBITS, N_LAYERS, scaled_to_physical,
                                     PARAM_DEFAULTS)
from inversion_engine       import run_inversion

RESULTS_DIR = os.path.join(src_dir, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# TRUE parameters — density in SCALED units (g/cc × 1e-3)
TRUE_PARAMS = np.array([
    4500.0,    # fault_x_loc (m)
    0.00020,   # rho_dense_scaled → 0.20 g/cc physical
    0.00015,   # rho_basin_scaled → 0.15 g/cc physical
    -1000.0,   # dense_depth_top (m)
])
NOISE_GRAVITY = 2.0   # mGal

print()
print("=" * 65)
print("  QUANTUM GEOPHYSICAL INVERSION v3 — PARAM SHAKTI")
print("=" * 65)
print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── Step 0: Diagnostics ───────────────────────────────────────────────────
print("\n[DIAG] Package versions:")
import numpy, scipy, harmonica, pennylane
print(f"  numpy={numpy.__version__}  scipy={scipy.__version__}  "
      f"harmonica={harmonica.__version__}  pennylane={pennylane.__version__}")

print("\n[DIAG] Forward model self-test (noise-free)...")
t0 = time.time()
_, g_t, K_t, U_t, Th_t, geo_t = run_complex_forward(
    fault_x_loc      = TRUE_PARAMS[0],
    rho_dense_scaled = TRUE_PARAMS[1],
    rho_basin_scaled = TRUE_PARAMS[2],
    dense_depth_top  = TRUE_PARAMS[3],
    noise_level=0., seed=0
)
phys_print = scaled_to_physical(TRUE_PARAMS)
print(f"  True params (physical): fault_x={TRUE_PARAMS[0]:.0f}m  "
      f"rho_dense={phys_print[1]:.3f}g/cc  "
      f"rho_basin={phys_print[2]:.3f}g/cc  "
      f"dense_top={TRUE_PARAMS[3]:.0f}m")
print(f"  Gravity range : {g_t.min():.2f} to {g_t.max():.2f} mGal")
print(f"  Forward time  : {time.time()-t0:.2f}s")

if abs(g_t).max() > 2000:
    print(f"  [FATAL] Gravity scale wrong ({abs(g_t).max():.0f} mGal). Aborting.")
    sys.exit(1)
print("  Gravity scale OK ✓")

print("\n[DIAG] Quantum circuit...")
w  = random_weights(seed=0)
ph = circuit_to_physical(w)
print(f"  Qubits={N_QUBITS}  Layers={N_LAYERS}  Weights={w.size}")
print(f"  Sample decode: fault_x={ph[0]:.1f}m  rho_d={ph[1]:.5f}  "
      f"rho_b={ph[2]:.5f}  top={ph[3]:.1f}m")
print("  Quantum circuit OK ✓")

# ── Step 1: Generate synthetic observations ───────────────────────────────
print("\n[STEP 1/2] Generating synthetic observations (true model + noise)...")
_, obs_gravity, K_obs, U_obs, Th_obs, _ = run_complex_forward(
    fault_x_loc      = TRUE_PARAMS[0],
    rho_dense_scaled = TRUE_PARAMS[1],
    rho_basin_scaled = TRUE_PARAMS[2],
    dense_depth_top  = TRUE_PARAMS[3],
    noise_level=NOISE_GRAVITY, seed=42
)
obs_radio = np.stack([K_obs, U_obs, Th_obs], axis=0)
print(f"  Gravity: {obs_gravity.min():.2f} to {obs_gravity.max():.2f} mGal "
      f"(noise={NOISE_GRAVITY} mGal)")

np.savez(os.path.join(RESULTS_DIR, "observed_data.npz"),
         true_gravity = g_t,
         true_params  = TRUE_PARAMS,
         obs_gravity  = obs_gravity,
         obs_radio    = obs_radio)

# ── Step 2: Inversion ─────────────────────────────────────────────────────
print("\n[STEP 2/2] Running multi-start quantum inversion...")
best_phys, mc_mean, mc_std, all_losses, all_params = run_inversion(
    obs_gravity, obs_radio,
    save_path   = os.path.join(RESULTS_DIR, "inversion_results.npz"),
    master_seed = 7,
)

# ── Final comparison (physical units) ─────────────────────────────────────
best_phys_print = scaled_to_physical(best_phys)
true_phys_print = scaled_to_physical(TRUE_PARAMS)
mc_mean_print   = scaled_to_physical(mc_mean)
mc_std_print    = scaled_to_physical(mc_std)

print()
print("=" * 70)
print("  FINAL COMPARISON (physical units)")
print("=" * 70)
labels = ['fault_x (m)', 'rho_dense (g/cc)', 'rho_basin (g/cc)', 'dense_top (m)']
print(f"  {'Parameter':<22} {'True':>10} {'Best':>10} {'MC mean':>10} {'MC ±σ':>10} {'Err%':>7}")
print(f"  {'-'*66}")
for i, lbl in enumerate(labels):
    tv  = true_phys_print[i]
    rv  = best_phys_print[i]
    err = abs(rv - tv) / (abs(tv) + 1e-12) * 100
    print(f"  {lbl:<22} {tv:>10.4f} {rv:>10.4f} "
          f"{mc_mean_print[i]:>10.4f} {mc_std_print[i]:>10.4f} {err:>6.1f}%")

print("=" * 70)
print(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("  Next: scp results/ to your PC and run: python src/plot_results.py")
