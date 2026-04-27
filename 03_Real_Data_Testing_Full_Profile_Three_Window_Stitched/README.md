# Real Data Testing: Full 30 km Profile

Contents:

- `qgeo_v3_realdata_full_profile_code/` - full-profile workflow using three
  independent 10 km inversions and stitching.
- `results_png/unified_three_window_plot.png` - best v5 unified 30 km stitched
  result.
- `results_png/window_00_10_individual_profile.png` - individual recovered
  geological/subsurface result for 0-10 km.
- `results_png/window_10_20_individual_profile.png` - individual recovered
  geological/subsurface result for 10-20 km.
- `results_png/window_20_30_individual_profile.png` - individual recovered
  geological/subsurface result for 20-30 km.
- `results_png/three_window_summary.json` - saved unified metrics and window
  parameters.

Typical local run:

```powershell
cd "qgeo_v3_realdata_full_profile_code"
python src\run_three_windows.py --maxiter 45 --popsize 9 --mc 12 --radio-weight 1.0
```

This is a stitched interpretation, not one globally coupled inversion. Each
10 km window has its own calibration and local model coordinates.
