#!/bin/bash
#SBATCH --qos=zen3_0512_devel
#SBATCH --job-name=gen_data_sample
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G

module load micromamba/latest
eval "$(micromamba shell hook --shell bash)"
micromamba activate shadows

./experiments/slurm/data_gen/gen_sample.sh vsc5 "$SLURM_ARRAY_TASK_ID"