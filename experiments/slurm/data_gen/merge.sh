#!/bin/bash
#SBATCH --qos=zen3_0512_devel
#SBATCH --job-name=merge_data
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=480G

module load micromamba/latest
eval "$(micromamba shell hook --shell bash)"
micromamba activate shadows

./experiments/slurm/data_gen/merge_data.sh vsc5 "$NUM_SAMPLES" "$NUM_MEAS"