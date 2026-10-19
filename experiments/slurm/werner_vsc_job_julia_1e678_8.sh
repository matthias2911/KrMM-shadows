#!/bin/bash
#SBATCH --qos=zen3_0512
#SBATCH --job-name=julia_1e678_8
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=480G
#SBATCH --exclusive

# Load micromamba and activate your environment
module load micromamba/latest
eval "$(micromamba shell hook --shell bash)"
micromamba activate shadows-vsc

module purge

export JULIA_NUM_THREADS=34
export OPENBLAS_NUM_THREADS=3

start=$SECONDS

./experiments/experiment.sh julia_vsc_1e6_8
./experiments/experiment.sh julia_vsc_1e7_8
./experiments/experiment.sh julia_vsc_1e8_8

elapsed=$((SECONDS - start))
echo "Wall time: ${elapsed}s"

echo "done"
