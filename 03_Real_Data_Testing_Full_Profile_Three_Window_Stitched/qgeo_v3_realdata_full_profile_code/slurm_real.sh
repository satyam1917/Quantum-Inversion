#!/bin/bash
#SBATCH --job-name=qgeo_real_best
#SBATCH --output=logs/real_best_%j.out
#SBATCH --error=logs/real_best_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

echo "Job=${SLURM_JOB_ID:-local} Node=${SLURMD_NODENAME:-unknown} Start=$(date)"
echo "Project directory: $(pwd)"

if [ -n "${QGEO_PYTHON:-}" ]; then
    PYTHON="$QGEO_PYTHON"
elif [ -x "$HOME/.conda/envs/qgeo_env/bin/python" ]; then
    PYTHON="$HOME/.conda/envs/qgeo_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="$(command -v python)"
fi

echo "Python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import numpy, scipy, matplotlib, harmonica; print('deps ok:', numpy.__version__, scipy.__version__, matplotlib.__version__, harmonica.__version__)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

MAXITER="${QGEO_MAXITER:-45}"
POPSIZE="${QGEO_POPSIZE:-9}"
MC="${QGEO_MC:-16}"

echo "Running calibrated real-data inversion..."
"$PYTHON" src/run_real_data.py \
    --csv real_data/real_testing_data.csv \
    --results-dir results \
    --window-start auto \
    --trend-degree 2 \
    --maxiter "$MAXITER" \
    --popsize "$POPSIZE" \
    --mc "$MC" \
    --seed 42

echo "Creating result plot..."
"$PYTHON" src/plot_real_results.py --results-dir results

echo "Done $(date)"
