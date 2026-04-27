# Quantum Inversion

Joint gravity and radiometric forward modeling and geophysical inversion using
quantum-assisted regression, synthetic quantum inversion, calibrated real-data
inversion, and full-profile stitched interpretation.

## Project Overview

This repository contains the final structured project for:

**Joint Gravity and Radiometric Forward Modeling and Geophysical Inversion**

The work is organized into four stages:

| Stage | Folder | Purpose |
|---|---|---|
| 0 | `00_Quantum_Regression` | Quantum linear and cubic regression on generated 100-point datasets |
| 1 | `01_Model_Training_and_Forward_Modeling` | Synthetic 10 km forward modeling and quantum-parameterized inversion |
| 2 | `02_Real_Data_Testing_10km_True_Inversion` | Best calibrated 10 km real-data inversion |
| 3 | `03_Real_Data_Testing_Full_Profile_Three_Window_Stitched` | Full 30 km profile from three stitched 10 km inversions |

The repository also contains an Overleaf-ready research report in:

```text
report/
```

## Main Results

### Stage 0: Quantum Regression

The regression stage uses small NumPy statevector simulators to create quantum
expectation-value features. Each experiment uses 100 generated data points with
an 80:20 train-test split.

| Model | Train R2 | Test R2 |
|---|---:|---:|
| Quantum linear regression | 0.997 | 0.996 |
| Quantum cubic regression | 0.996 | 0.994 |

### Stage 1: Synthetic Geophysical Inversion

The synthetic model represents a 10 km profile containing host rock, dense body,
faulted basin, and soil. Gravity is computed using a 2.5D prism model, and K/U/Th
radiometric responses are generated from lithological endmembers.

Recovered synthetic parameter errors:

| Parameter | Error |
|---|---:|
| Fault position | 6.2 percent |
| Dense-body density contrast | 3.4 percent |
| Basin density contrast | 3.2 percent |
| Dense-body top depth | 4.0 percent |

### Stage 2: Best 10 km Real-Data Inversion

The real-data workflow uses robust regional-trend removal, automatic window
selection, scale-offset calibration, real-style radiometric enrichment terms,
and Monte Carlo uncertainty estimation.

| Channel | R2 | RMSE |
|---|---:|---:|
| Gravity | 0.989 | 1.267 mGal |
| K | 0.951 | 0.238 |
| U | 0.929 | 0.633 |
| Th | 0.891 | 2.541 |

### Stage 3: Full 30 km Stitched Profile

The full profile is interpreted using three independent 10 km inversions stitched
across the 30 km line. This is a stitched interpretation, not a single globally
coupled inversion.

| Channel | R2 |
|---|---:|
| Gravity | 0.965 |
| K | 0.883 |
| U | 0.876 |
| Th | 0.866 |

## Repository Structure

```text
Quantum-Inversion/
  00_Quantum_Regression/
    Linear Regression/
    Cubic Regression/
  01_Model_Training_and_Forward_Modeling/
  02_Real_Data_Testing_10km_True_Inversion/
  03_Real_Data_Testing_Full_Profile_Three_Window_Stitched/
  report/
```

## Running Locally

### Quantum Linear Regression

```powershell
cd "00_Quantum_Regression\Linear Regression"
python -m pip install -r requirements.txt
python src\run_quantum_linear_regression.py --regenerate
```

### Quantum Cubic Regression

```powershell
cd "00_Quantum_Regression\Cubic Regression"
python -m pip install -r requirements.txt
python src\run_quantum_cubic_regression.py --regenerate
```

### Best 10 km Real-Data Inversion

```powershell
cd "02_Real_Data_Testing_10km_True_Inversion\qgeo_v3_realdata_10km_code"
python -m pip install -r requirements.txt
python src\run_real_data.py --maxiter 45 --popsize 9 --mc 16
python src\plot_real_results.py
```

### Full 30 km Three-Window Inversion

```powershell
cd "03_Real_Data_Testing_Full_Profile_Three_Window_Stitched\qgeo_v3_realdata_full_profile_code"
python -m pip install -r requirements.txt
python src\run_three_windows.py --maxiter 45 --popsize 9 --mc 12 --radio-weight 1.0
```

## Param Shakti Execution

Each computational stage includes Slurm scripts. Typical usage:

```bash
cd "$SCRATCH/Quantum-Inversion/03_Real_Data_Testing_Full_Profile_Three_Window_Stitched/qgeo_v3_realdata_full_profile_code"
export QGEO_MAXITER=45
export QGEO_POPSIZE=9
export QGEO_MC=12
export QGEO_RADIO_WEIGHT=1.0
JOBID=$(sbatch --parsable slurm_three_windows.sh)
echo "Submitted job $JOBID"
tail -f "logs/three_windows_${JOBID}.out"
```

## Report

The complete research-style report is in:

```text
report/main.tex
```

Upload the `report/` folder to Overleaf and compile with `pdfLaTeX`.

## Notes

- Density contrasts from real-data inversion are calibrated effective contrasts.
- Radiometric parameters are shallow response-shape controls, not deep density
  parameters.
- The 30 km result is a stitched interpretation from three independent windows.
- The quantum workflows are classical simulations of quantum circuits/feature
  maps and are designed to be portable to future quantum backends.

