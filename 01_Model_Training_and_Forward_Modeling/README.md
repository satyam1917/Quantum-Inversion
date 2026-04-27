# Model Training and Forward Modeling

Contents:

- `qgeo_v3_training_code/` - original synthetic training package from
  `qgeo_v3.zip`.
- `results_png/training_results_v3.png` - final v3 training/result figure.

Typical local run:

```powershell
cd "qgeo_v3_training_code"
python src\run_inversion.py
python src\plot_results.py
```

The training workflow uses the synthetic 10 km forward model and compares the
true geological model against the recovered geological model.
