#!/bin/bash
#SBATCH --qos=zen3_0512_devel
#SBATCH --job-name=test
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=00:10:00
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

module load cmake/3.24.3-aocc-4.0.0-uy23dkp
module load aocc/4.1.0-gcc-12.2.0-rir6635
module load openmpi/4.1.4-aocc-4.0.0-x4jtiii
module load openblas/0.3.21-aocc-4.0.0-nmupasv

export CC=clang
export CXX=clang++
export LD_LIBRARY_PATH="${LD_RUN_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" # otherwise binary cannot find openblas.so path

# ./scripts/configure.sh
# ./scripts/build.sh

# DEBUG:
# echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
# echo "LD_RUN_PATH=$LD_RUN_PATH"
# which clang++
# which mpicc
# clang++ --version

./experiments/experiment.sh werner_vsc


echo "done"
