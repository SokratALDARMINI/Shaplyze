#!/bin/bash
#SBATCH --job-name=SH_1
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=logs/slurm_1-%A_%a.out
#SBATCH --error=logs/slurm_1-%A_%a.err

echo "Starting SLURM job array task ${SLURM_ARRAY_TASK_ID}"

module purge
module load anaconda
eval "$(conda shell.bash hook)"
conda activate p312_ShapleyValue

python -u benchmarking_cluster.py --ID ${SLURM_ARRAY_TASK_ID}
