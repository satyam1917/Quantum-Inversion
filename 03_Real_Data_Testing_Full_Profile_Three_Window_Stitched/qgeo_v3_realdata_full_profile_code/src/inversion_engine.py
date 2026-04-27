# =============================================================================
# inversion_engine.py  (v3 CORRECTED)
# =============================================================================

import numpy as np
from scipy.optimize import minimize
import os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complex_forward_model import run_complex_forward
from quantum_ansatz import (
    circuit_to_physical, random_weights, lhs_weights,
    weights_from_physical, get_weight_shape,
    normalize, PARAM_DEFAULTS, PARAM_NAMES, scaled_to_physical
)

N_STARTS         = 6
MAX_ITER_PER_RUN = 200
N_MC_SAMPLES     = 10
MC_NOISE_FACTOR  = 0.5
GRAVITY_WEIGHT   = 1.0
RADIO_WEIGHT     = 0.12
REG_WEIGHT       = 0.03


def build_cost(obs_gravity, obs_radio):
    weight_shape = get_weight_shape()
    g_std = float(np.std(obs_gravity)) + 1e-8
    r_std = np.std(obs_radio, axis=1, keepdims=True) + 1e-8

    def cost(w_flat):
        weights = w_flat.reshape(weight_shape)
        try:
            phys = circuit_to_physical(weights)
        except Exception:
            return 1e8
        try:
            _, g_p, K_p, U_p, Th_p, _ = run_complex_forward(
                fault_x_loc      = phys[0],
                rho_dense_scaled = phys[1],
                rho_basin_scaled = phys[2],
                dense_depth_top  = phys[3],
                noise_level      = 0.,
            )
        except Exception:
            return 1e8

        mg  = float(np.mean(((g_p - obs_gravity) / g_std) ** 2))
        rp  = np.stack([K_p, U_p, Th_p], axis=0)
        mr  = float(np.mean(((rp - obs_radio) / r_std) ** 2))
        reg = float(np.sum((normalize(phys) - 0.5) ** 2))
        return GRAVITY_WEIGHT * mg + RADIO_WEIGHT * mr + REG_WEIGHT * reg

    return cost, int(np.prod(weight_shape))


def _single_run(obs_gravity, obs_radio, w0, run_id):
    cost_fn, n_w = build_cost(obs_gravity, obs_radio)
    t0 = time.time()
    calls = [0]

    def cost_logged(w):
        calls[0] += 1
        v = cost_fn(w)
        if calls[0] % 40 == 0:
            p    = circuit_to_physical(w.reshape(get_weight_shape()))
            phys = scaled_to_physical(p)
            print(f"      call {calls[0]:4d} | loss={v:.5f} | "
                  f"fault_x={p[0]:.0f}m  "
                  f"rho_dense={phys[1]:.3f}g/cc  "
                  f"rho_basin={phys[2]:.3f}g/cc  "
                  f"dense_top={p[3]:.0f}m", flush=True)
        return v

    try:
        res   = minimize(cost_logged, w0.ravel(), method='L-BFGS-B',
                         options={'maxiter': MAX_ITER_PER_RUN,
                                  'ftol': 1e-11, 'gtol': 1e-8,
                                  'maxfun': MAX_ITER_PER_RUN * (n_w+1),
                                  'disp': False})
        final_w    = res.x.reshape(get_weight_shape())
        final_phys = circuit_to_physical(final_w)
        final_loss = float(res.fun)
    except Exception as e:
        print(f"  [WARN] Run {run_id} crashed: {e}")
        traceback.print_exc()
        final_phys = PARAM_DEFAULTS.copy()
        final_loss = 1e8

    phys_print = scaled_to_physical(final_phys)
    print(f"  Run {run_id} | loss={final_loss:.5f} | "
          f"fault_x={final_phys[0]:.1f}m  "
          f"rho_dense={phys_print[1]:.3f}g/cc  "
          f"rho_basin={phys_print[2]:.3f}g/cc  "
          f"dense_top={final_phys[3]:.1f}m | {time.time()-t0:.0f}s")
    return final_loss, final_phys


def run_inversion(obs_gravity, obs_radio, save_path=None, master_seed=0):
    np.random.seed(master_seed)
    print()
    print("=" * 65)
    print("  QUANTUM INVERSION v3  (LHS multi-start + MC uncertainty)")
    print("=" * 65)
    print(f"  N_STARTS={N_STARTS}  MAX_ITER={MAX_ITER_PER_RUN}  N_MC={N_MC_SAMPLES}")

    lhs_init  = lhs_weights(N_STARTS - 1, seed=master_seed)
    warm_init = weights_from_physical(PARAM_DEFAULTS, noise_std=0.4)

    all_losses, all_params = [], []
    best_loss  = np.inf
    best_phys  = None

    for s in range(N_STARTS):
        label = "warm-start" if s == 0 else f"LHS-{s}"
        print(f"\n--- Start {s+1}/{N_STARTS} ({label}) ---")
        w0   = warm_init if s == 0 else lhs_init[s-1]
        loss, phys = _single_run(obs_gravity, obs_radio, w0, s+1)
        all_losses.append(loss)
        all_params.append(phys)
        if loss < best_loss:
            best_loss = loss
            best_phys = phys.copy()
            print(f"  *** New best (loss={best_loss:.5f}) ***")

    # Monte Carlo uncertainty
    print(f"\n--- Monte Carlo uncertainty ({N_MC_SAMPLES} samples) ---")
    noise_std = float(np.std(obs_gravity)) * MC_NOISE_FACTOR
    mc_params = []
    for mc in range(N_MC_SAMPLES):
        rng       = np.random.default_rng(master_seed + 1000 + mc)
        g_pert    = obs_gravity + rng.normal(0, noise_std, obs_gravity.shape)
        r_pert    = obs_radio   + rng.normal(0, noise_std * 0.05, obs_radio.shape)
        w0        = weights_from_physical(best_phys, noise_std=0.2)
        cfn, _    = build_cost(g_pert, r_pert)
        try:
            res   = minimize(cfn, w0.ravel(), method='L-BFGS-B',
                             options={'maxiter': 80, 'ftol': 1e-10, 'disp': False})
            mc_ph = circuit_to_physical(res.x.reshape(get_weight_shape()))
        except Exception:
            mc_ph = best_phys.copy()
        mc_params.append(mc_ph)
        if (mc+1) % 5 == 0:
            print(f"  MC {mc+1}/{N_MC_SAMPLES} done", flush=True)

    mc_arr  = np.array(mc_params)
    mc_mean = mc_arr.mean(axis=0)
    mc_std  = mc_arr.std(axis=0)

    # Summary in physical units
    bp    = scaled_to_physical(best_phys)
    mcm   = scaled_to_physical(mc_mean)
    mcs   = scaled_to_physical(mc_std)

    print()
    print("=" * 65)
    print("  RESULT SUMMARY (physical units)")
    print("=" * 65)
    print(f"  {'Parameter':<22} {'Best':>12} {'MC mean':>12} {'MC ±1σ':>12}")
    print(f"  {'-'*60}")
    pnames = ['fault_x (m)', 'rho_dense (g/cc)', 'rho_basin (g/cc)', 'dense_top (m)']
    for i, nm in enumerate(pnames):
        print(f"  {nm:<22} {bp[i]:>12.4f} {mcm[i]:>12.4f} {mcs[i]:>12.4f}")

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(save_path,
                 best_phys   = best_phys,
                 best_loss   = np.array([best_loss]),
                 all_losses  = np.array(all_losses),
                 all_params  = np.array(all_params),
                 mc_params   = mc_arr,
                 mc_mean     = mc_mean,
                 mc_std      = mc_std,
                 obs_gravity = obs_gravity,
                 obs_radio   = obs_radio)
        print(f"\n  Saved -> {save_path}")

    return best_phys, mc_mean, mc_std, np.array(all_losses), np.array(all_params)
