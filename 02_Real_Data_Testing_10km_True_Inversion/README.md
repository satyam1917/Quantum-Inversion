# Real Data Testing: 10 km True Inversion

Contents:

- `qgeo_v3_realdata_10km_code/` - calibrated real-data workflow for the original
  single 10 km window inversion.
- `results_png/real_data_best_plot.png` - best 10 km real-data result plot.
- `results_png/real_data_best_summary.json` - saved metrics and recovered
  parameters.
- `results_png/real_data_best_results.npz` - saved numerical arrays for plotting
  or later analysis.

Typical local run:

```powershell
cd "qgeo_v3_realdata_10km_code"
python src\run_real_data.py --maxiter 45 --popsize 9 --mc 16
python src\plot_real_results.py
```

This stage selects and inverts the best 10 km window from the real profile,
without stitching the full 30 km line.
