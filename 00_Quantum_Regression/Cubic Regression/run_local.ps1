$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import numpy, matplotlib; print('dependencies ok')"
python src\run_quantum_cubic_regression.py --regenerate

Write-Host "Quantum cubic regression complete."
Write-Host "Plot: results\quantum_cubic_regression.png"
Write-Host "Summary: results\quantum_cubic_summary.json"
