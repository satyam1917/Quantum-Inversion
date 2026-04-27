# Quantum Geophysical Inversion v3

## What's new vs v2

| Issue | v2 | v3 |
|---|---|---|
| Gravity scale | ~10^6 mGal (bug) | Realistic ±200 mGal |
| Cause | Air prisms included (-2670 kg/m³) | Air prisms excluded from calc |
| Multi-start diversity | All starts clustered | Latin Hypercube Sampling (LHS) |
| Uncertainty | None | Monte Carlo ± 1σ on all parameters |
| Real data | Not supported | Full pipeline in real_data_pipeline.py |

## Project structure

```
qgeo_v3/
├── src/
│   ├── complex_forward_model.py  ← FIXED: realistic Bouguer gravity
│   ├── quantum_ansatz.py         ← PQC: 8 qubits, LHS initialisation
│   ├── inversion_engine.py       ← LHS multi-start + MC uncertainty
│   ├── run_inversion.py          ← master runner (Slurm entry point)
│   ├── plot_results.py           ← 5-row results figure with error bars
│   └── real_data_pipeline.py     ← CSV loader, Bouguer corr, real inversion
├── real_data/                    ← put your CSV files here
├── results/
├── logs/
└── slurm_job.sh
```

## Run on Param Shakti (synthetic data)

```bash
# Upload
scp -r qgeo_v3/ 22ex23013@paramshakti.iitkgp.ac.in:/scratch/22ex23013/

# On SSH terminal
cd /scratch/22ex23013/qgeo_v3
sbatch slurm_job.sh
tail -f logs/qgeo_<JOBID>.out
```

## Run on real data

### Step 1 — Prepare your CSV files

Gravity CSV needs columns: `x` (metres along profile), `bouguer_anomaly` (mGal).
If you only have raw gravity: add `elevation` column and set `apply_bouguer=True`.

Radiometric CSV needs: `x`, `k` (%), `u` (ppm), `th` (ppm).

### Step 2 — Upload data to Param Shakti

```bash
scp your_gravity.csv  22ex23013@paramshakti.iitkgp.ac.in:/scratch/22ex23013/qgeo_v3/real_data/
scp your_radio.csv    22ex23013@paramshakti.iitkgp.ac.in:/scratch/22ex23013/qgeo_v3/real_data/
```

### Step 3 — Run real data inversion

Create a new script `run_real.py`:
```python
from real_data_pipeline import run_real_data_inversion
run_real_data_inversion(
    gravity_csv   = "real_data/your_gravity.csv",
    radio_csv     = "real_data/your_radio.csv",
    apply_bouguer = False,   # True if raw gravity provided
    lat_deg       = 22.3,    # your survey latitude
    remove_trend  = True,
    results_dir   = "results",
)
```

### Step 4 — Transfer and plot

```powershell
scp -r 22ex23013@paramshakti.iitkgp.ac.in:/scratch/22ex23013/qgeo_v3/results/ .\results\
python src\plot_results.py
```

## Acknowledgement

> "This work used the Supercomputing facility of IIT Kharagpur established under National
> Supercomputing Mission (NSM), Government of India and supported by Centre for Development
> of Advanced Computing (CDAC), Pune."
