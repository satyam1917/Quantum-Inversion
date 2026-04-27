$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import numpy, matplotlib; print('dependencies ok')"
python src\run_quantum_linear_regression.py --regenerate

Write-Host "Quantum linear regression complete."
Write-Host "Plot: results\quantum_linear_regression.png"
Write-Host "Summary: results\quantum_linear_summary.json"
