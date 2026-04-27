# Quantum Cubic Regression

This folder generates 100 synthetic cubic data points, splits them into
80 percent training and 20 percent testing, and fits a quantum feature-map
regressor using a small NumPy statevector simulator.

No external quantum framework is required. The model encodes `x`, `x^2`, and
`x^3` into a four-qubit feature map, measures quantum expectation features, and
solves the regularized inverse problem for the readout weights.

Local run:

```powershell
python src\run_quantum_cubic_regression.py --regenerate
```

Param Shakti run:

```bash
sbatch slurm_cubic_regression.sh
```

Outputs:

- `data/cubic_regression_data.csv`
- `results/quantum_cubic_regression.png`
- `results/quantum_cubic_summary.json`
- `results/quantum_cubic_model.npz`
