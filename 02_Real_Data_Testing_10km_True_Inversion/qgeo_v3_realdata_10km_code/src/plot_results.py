# =============================================================================
# plot_results.py  (v3 CORRECTED) — run LOCALLY after scp
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)
from complex_forward_model import (run_complex_forward, X_COORDS, Z_COORDS,
                                    X_MIN, X_MAX, Z_MIN, TOPOGRAPHY_Z, _f_topo)
from quantum_ansatz import scaled_to_physical

CMAP_GEO   = ListedColormap(['#4b0082','#006400','#e8c840','#8B4513'])
GEO_LABELS = ['Host Rock','Dense Body','Basin','Soil']

# True params in SCALED units
TRUE_PARAMS = np.array([4500.0, 0.00020, 0.00015, -1000.0])


def _geo_panel(ax, geo, title):
    geo_plot = np.clip(geo - 1, 0, 3)
    ax.imshow(geo_plot, extent=[X_MIN/1000, X_MAX/1000, Z_MIN/1000, 0],
              origin='lower', aspect='auto', cmap=CMAP_GEO,
              interpolation='none', vmin=0, vmax=3)
    ax.plot(X_COORDS/1000, TOPOGRAPHY_Z/1000, 'k-', lw=1.5, alpha=0.7)
    ax.set_ylabel('Elevation (km)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(Z_MIN/1000, 0.1)
    handles = [Patch(color=CMAP_GEO(i/3), label=GEO_LABELS[i]) for i in range(4)]
    ax.legend(handles=handles, loc='lower right', fontsize=7, ncol=2)


def plot_all(results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(src_dir, '..', 'results')

    inv_path = os.path.join(results_dir, 'inversion_results.npz')
    if not os.path.exists(inv_path):
        print(f"[ERROR] {inv_path} not found.")
        return

    inv     = np.load(inv_path)
    best    = inv['best_phys']         # scaled units
    obs_g   = inv['obs_gravity']
    obs_r   = inv['obs_radio']
    all_l   = inv['all_losses']
    all_p   = inv['all_params']
    mc_p    = inv['mc_params']
    mc_mean = inv['mc_mean']
    mc_std  = inv['mc_std']

    obs_d   = np.load(os.path.join(results_dir, 'observed_data.npz'))
    true_g  = obs_d.get('true_gravity', None)

    x_obs = np.linspace(X_MIN, X_MAX, len(obs_g))

    # Forward models using SCALED params
    _, g_rec, K_r, U_r, Th_r, geo_rec = run_complex_forward(
        fault_x_loc=best[0], rho_dense_scaled=best[1],
        rho_basin_scaled=best[2], dense_depth_top=best[3], noise_level=0.)
    _, g_true, K_t, U_t, Th_t, geo_true = run_complex_forward(
        fault_x_loc=TRUE_PARAMS[0], rho_dense_scaled=TRUE_PARAMS[1],
        rho_basin_scaled=TRUE_PARAMS[2], dense_depth_top=TRUE_PARAMS[3],
        noise_level=0.)

    # Physical units for labels
    bp = scaled_to_physical(best)
    tp = scaled_to_physical(TRUE_PARAMS)
    mm = scaled_to_physical(mc_mean)
    ms = scaled_to_physical(mc_std)

    # MC gravity envelope
    g_mc_all = []
    for mc_ph in mc_p[:min(10, len(mc_p))]:
        try:
            _, gm, _, _, _, _ = run_complex_forward(
                fault_x_loc=mc_ph[0], rho_dense_scaled=mc_ph[1],
                rho_basin_scaled=mc_ph[2], dense_depth_top=mc_ph[3],
                noise_level=0.)
            g_mc_all.append(gm)
        except Exception:
            pass

    # ── Figure ──────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        "Quantum Geophysical Inversion v3 — Results\n"
        f"Recovered: fault_x={bp[0]:.0f}m  rho_dense={bp[1]:.3f}g/cc  "
        f"rho_basin={bp[2]:.3f}g/cc  dense_top={bp[3]:.0f}m\n"
        f"True:      fault_x={tp[0]:.0f}m  rho_dense={tp[1]:.3f}g/cc  "
        f"rho_basin={tp[2]:.3f}g/cc  dense_top={tp[3]:.0f}m",
        fontsize=12, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(5, 2, hspace=0.50, wspace=0.35)

    # Row 0: geology
    _geo_panel(fig.add_subplot(gs[0,0]), geo_true, "TRUE Geological Model")
    _geo_panel(fig.add_subplot(gs[0,1]), geo_rec,  "RECOVERED Geological Model")

    # Row 1: multi-start scatter + error bars
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_eb = fig.add_subplot(gs[1, 1])

    # Convert all_p to physical for scatter
    all_p_phys = np.array([scaled_to_physical(p) for p in all_p])
    ln  = (all_l - all_l.min()) / (all_l.max() - all_l.min() + 1e-12)
    sc  = ax_sc.scatter(all_p_phys[:,0], all_p_phys[:,1], c=ln,
                         cmap='RdYlGn_r', s=130, zorder=3, edgecolors='k', lw=0.8)
    ax_sc.axvline(tp[0], color='blue',  ls='--', lw=1.5, label=f"True fault_x={tp[0]:.0f}m")
    ax_sc.axhline(tp[1], color='green', ls='--', lw=1.5, label=f"True rho_dense={tp[1]:.3f}")
    ax_sc.scatter([bp[0]], [bp[1]], marker='*', c='gold', s=350, zorder=5,
                   edgecolors='k', lw=1, label='Best')
    plt.colorbar(sc, ax=ax_sc, label='Loss (green=best)')
    ax_sc.set_xlabel('fault_x_loc (m)'); ax_sc.set_ylabel('rho_dense (g/cc)')
    ax_sc.set_title('Multi-Start: LHS diversity', fontweight='bold')
    ax_sc.legend(fontsize=8)

    # Normalised bar: true vs recovered vs MC
    norm_b = bp / (tp + 1e-12)
    norm_m = mm / (tp + 1e-12)
    norm_s = ms / (np.abs(tp) + 1e-12)
    x4     = np.arange(4)
    ax_eb.bar(x4-0.25, np.ones(4), 0.22, label='True (=1)', color='steelblue', alpha=0.8)
    ax_eb.bar(x4,      norm_b,     0.22, label='Best',      color='tomato',    alpha=0.8)
    ax_eb.errorbar(x4+0.25, norm_m, yerr=norm_s, fmt='o',
                    color='darkorange', capsize=5, lw=2, label='MC mean ±1σ')
    ax_eb.axhline(1., color='k', ls=':', lw=0.8)
    ax_eb.set_xticks(x4)
    ax_eb.set_xticklabels(['fault_x\n(m)', 'rho_dense\n(g/cc)', 'rho_basin\n(g/cc)', 'dense_top\n(m)'])
    ax_eb.set_ylabel('Ratio (Recovered / True)')
    ax_eb.set_title('Parameter Recovery with MC Uncertainty', fontweight='bold')
    ax_eb.legend(fontsize=8)

    # Row 2: gravity fit (full width)
    ax_g = fig.add_subplot(gs[2, :])
    ax_g.plot(x_obs/1000, obs_g,  'k-',  lw=1.0, alpha=0.6, label='Observed (noisy)')
    ax_g.plot(x_obs/1000, g_true, 'b--', lw=2.0, alpha=0.9, label='True (noise-free)')
    ax_g.plot(x_obs/1000, g_rec,  'r-',  lw=2.0, alpha=0.9, label='Recovered')
    if g_mc_all:
        gma = np.array(g_mc_all)
        ax_g.fill_between(x_obs/1000, gma.min(0), gma.max(0),
                           alpha=0.2, color='orange', label='MC envelope')
    ax_g.set_xlabel('Distance (km)'); ax_g.set_ylabel('Gravity anomaly (mGal)')
    ax_g.set_title('Gravity Data Fit (realistic mGal scale)', fontweight='bold')
    ax_g.legend(fontsize=9)
    ax_g2 = ax_g.twinx()
    ax_g2.fill_between(x_obs/1000, g_rec - g_true, 0, color='orange', alpha=0.3)
    ax_g2.set_ylabel('Residual (mGal)', color='darkorange')
    ax_g2.tick_params(axis='y', labelcolor='darkorange')

    # Rows 3-4: radiometric
    for ax, obs, true_ch, rec_ch, lbl, unit in [
        (fig.add_subplot(gs[3,0]), obs_r[0], K_t, K_r, 'K', '%'),
        (fig.add_subplot(gs[3,1]), obs_r[1], U_t, U_r, 'U', 'ppm'),
        (fig.add_subplot(gs[4,0]), obs_r[2], Th_t, Th_r, 'Th', 'ppm'),
    ]:
        ax.plot(x_obs/1000, obs,     'k-',  lw=1, alpha=0.6, label=f'Obs {lbl}')
        ax.plot(x_obs/1000, true_ch, 'b--', lw=2,            label=f'True {lbl}')
        ax.plot(x_obs/1000, rec_ch,  'r-',  lw=2,            label=f'Recovered {lbl}')
        ax.set_xlabel('Distance (km)'); ax.set_ylabel(f'{lbl} ({unit})')
        ax.set_title(f'Radiometric Fit: {lbl}', fontweight='bold')
        ax.legend(fontsize=8)

    # Summary table
    ax_tbl = fig.add_subplot(gs[4, 1]); ax_tbl.axis('off')
    rows = []
    for i, nm in enumerate(['fault_x (m)', 'rho_dense (g/cc)', 'rho_basin (g/cc)', 'dense_top (m)']):
        err = abs(bp[i] - tp[i]) / (abs(tp[i]) + 1e-12) * 100
        rows.append([nm, f"{tp[i]:.4f}", f"{bp[i]:.4f}",
                     f"{mm[i]:.4f}±{ms[i]:.4f}", f"{err:.1f}%"])
    tbl = ax_tbl.table(cellText=rows,
                        colLabels=['Parameter','True','Best','MC mean±σ','Error%'],
                        loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.8)
    ax_tbl.set_title('Summary (physical units)', fontweight='bold')

    out = os.path.join(results_dir, 'inversion_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Plot saved -> {out}")
    plt.show()


if __name__ == "__main__":
    plot_all()
