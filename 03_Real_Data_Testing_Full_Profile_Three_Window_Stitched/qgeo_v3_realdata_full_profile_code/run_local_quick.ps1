$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import numpy, scipy, matplotlib, harmonica; print('dependencies ok')"
python src\run_real_data.py --quick
python src\plot_real_results.py
python src\run_three_windows.py --quick

Write-Host "Quick run complete."
Write-Host "Single-window plot: results\real_data_best_plot.png"
Write-Host "Three-window plot: results\unified_three_window_plot.png"
