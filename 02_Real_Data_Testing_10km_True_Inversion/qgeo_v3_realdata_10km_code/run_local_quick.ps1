$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import numpy, scipy, matplotlib, harmonica; print('dependencies ok')"
python src\run_real_data.py --quick
python src\plot_real_results.py

Write-Host "Quick run complete. Plot: results\real_data_best_plot.png"
