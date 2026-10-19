#!/bin/bash
#SBATCH --qos=zen3_0512_devel
#SBATCH --job-name=gen_data
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G

# Load micromamba and activate your environment
module load micromamba/latest
eval "$(micromamba shell hook --shell bash)"
micromamba activate shadows

./experiments/gen_data.sh vsc5

echo "done"
