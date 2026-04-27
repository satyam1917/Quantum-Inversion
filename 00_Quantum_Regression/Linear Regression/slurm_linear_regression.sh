#!/bin/bash
#SBATCH --job-name=qreg_linear
#SBATCH --output=logs/qreg_linear_%j.out
#SBATCH --error=logs/qreg_linear_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results data

echo "Job=${SLURM_JOB_ID:-local} Node=${SLURMD_NODENAME:-unknown} Start=$(date)"

if [ -n "${QREG_PYTHON:-}" ]; then
    PYTHON="$QREG_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="$(command -v python)"
fi

echo "Python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import numpy, matplotlib; print('deps ok')"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" src/run_quantum_linear_regression.py \
    --n-points "${QREG_N_POINTS:-100}" \
    --train-ratio "${QREG_TRAIN_RATIO:-0.8}" \
    --seed "${QREG_SEED:-42}" \
    --alpha "${QREG_ALPHA:-1e-6}" \
    --regenerate

echo "Done $(date)"
