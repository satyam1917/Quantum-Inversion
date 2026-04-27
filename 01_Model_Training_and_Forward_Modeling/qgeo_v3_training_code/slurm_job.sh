#!/bin/bash
#SBATCH --job-name=qgeo_v3
#SBATCH --output=/scratch/22ex23013/qgeo_v3/logs/qgeo_%j.out
#SBATCH --error=/scratch/22ex23013/qgeo_v3/logs/qgeo_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=22ex23013@kgpian.iitkgp.ac.in

echo "Job=$SLURM_JOB_ID  Node=$SLURMD_NODENAME  Start=$(date)"
cd /scratch/22ex23013/qgeo_v3 || { echo "FATAL: dir not found"; exit 1; }
mkdir -p logs results

PYTHON=/home/22ex23013/.conda/envs/qgeo_env/bin/python
$PYTHON --version || { echo "FATAL: Python missing"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON src/run_inversion.py
CODE=$?
echo "Done $(date) | exit=$CODE"
exit $CODE
