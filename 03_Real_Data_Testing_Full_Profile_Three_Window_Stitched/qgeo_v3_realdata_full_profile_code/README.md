# Quantum Geophysical Inversion v3 - Real Data Best Package

This package keeps the original v3 synthetic model files, and adds a real-data
runner designed for the supplied `real_testing_data.csv`.

Current real-data default: calibrated v5 model.

## Why the previous real-data plot failed

The old real-data script sent the real residual anomaly directly into the
synthetic inversion engine. That made three bad assumptions:

1. Synthetic gravity amplitude and sign were treated as absolute.
2. Synthetic K/U/Th levels were treated as absolute.
3. The real profile direction and radiometric polarity were assumed to match the
   synthetic model.
4. The synthetic K/U/Th model was only a left/right step, while the real profile
   contains a localized shallow radiometric high near 1-2.5 km.

For field data, Bouguer residuals and radiometrics need survey calibration before
being compared with a synthetic forward model. The new runner fits the geological
shape while estimating per-survey scale and offset terms for gravity, K, U, and
Th. It also scans possible 10 km windows and profile orientation before the final
optimization. The v5 real-data path expands the depth domain from 2.5 km to
3.5 km and adds two shallow radiometric enrichment lobes plus a mild trend term
to the synthetic forward model, so the synthetic training geometry is closer to
the real 30 km profile.

## Main files

- `src/run_real_data.py` - calibrated real-data inversion entry point.
- `src/real_data_engine.py` - preprocessing, auto-window search, optimization,
  Monte Carlo uncertainty, real-style radiometric proxy, and result saving.
- `src/plot_real_results.py` - publication-style result figure.
- `slurm_real.sh` - Param Shakti Slurm batch script.
- `real_data/real_testing_data.csv` - supplied real testing data.
- `requirements.txt` - real-data dependencies.
- `requirements-quantum-optional.txt` - optional PennyLane dependencies for the
  original synthetic quantum ansatz scripts.

## Local quick smoke test

From PowerShell:

```powershell
cd "C:\Users\satya\OneDrive\Documents\New project\qgeo_v3_realdata_best_v5"
.\run_local_quick.ps1
```

The smoke test uses fewer iterations. For better local results, run:

```powershell
python src\run_real_data.py --maxiter 45 --popsize 9 --mc 16
python src\plot_real_results.py
```

## Three-window full-profile inversion

The real-data forward model is still a 10 km local model. To cover the full
30 km profile, use three independent windows and stitch the interpreted results:

```powershell
python src\run_three_windows.py --maxiter 45 --popsize 9 --mc 12 --radio-weight 1.0
```

This writes:

- `results/window_00_10/`
- `results/window_10_20/`
- `results/window_20_30/`
- `results/unified_three_window_plot.png`
- `results/three_window_summary.json`
- `results/window_00_10_individual_profile.png`
- `results/window_10_20_individual_profile.png`
- `results/window_20_30_individual_profile.png`

This is feasible as a stitched interpretation. It is not a single globally
coupled inversion, because each window has its own regional residual, calibration
scale, local coordinates, and boundary conditions.

## Param Shakti run

Upload the ZIP to `/scratch/<username>/`, unzip it, and submit:

```bash
cd $SCRATCH/qgeo_v3_realdata_best_v5
sbatch slurm_real.sh
```

For the full 30 km three-window product:

```bash
cd $SCRATCH/qgeo_v3_realdata_best_v5
sbatch slurm_three_windows.sh
```

For later windows, K/U/Th are harder to fit than gravity. The Slurm script uses
`QGEO_RADIO_WEIGHT=1.0` by default for a better full-profile radiometric fit; you
can raise it to `1.2` if you want to prioritize radiometrics further.

The final files are written to:

- `results/real_data_best_results.npz`
- `results/real_data_best_summary.json`
- `results/real_data_best_plot.png`

## Notes

The reported density contrasts are effective contrasts under the calibrated
survey model. If the saved JSON shows a negative gravity scale, it means the
observed residual anomaly polarity is opposite to the synthetic `g_z` convention
after regional-trend removal; the plot uses the calibrated model so the data and
forward response are compared consistently.

The radiometric peak parameters are surface-response shape parameters, not deep
density parameters. They are included because K/U/Th data are shallow and cannot
be represented well by the old deep-only basin/host step model.
