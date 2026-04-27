# Quantum Linear Regression

This folder generates 100 synthetic linear data points, splits them into
80 percent training and 20 percent testing, and fits a quantum feature-map
regressor using a small NumPy statevector simulator.

No external quantum framework is required. The model encodes each input into a
two-qubit feature map, measures quantum expectation features, and solves the
regularized inverse problem for the readout weights.

Local run:

```powershell
python src\run_quantum_linear_regression.py --regenerate
```

Param Shakti run:

```bash
sbatch slurm_linear_regression.sh
```

Outputs:

- `data/linear_regression_data.csv`
- `results/quantum_linear_regression.png`
- `results/quantum_linear_summary.json`
- `results/quantum_linear_model.npz`
