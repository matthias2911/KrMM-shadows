#!/bin/bash
NUM_SAMPLES=4
NUM_MEAS=25000000

# submit array job: task ids 0..NUM_SAMPLES-1
array_jobid=$(sbatch --parsable --array=0-$((NUM_SAMPLES-1)) experiments/slurm/data_gen/sample.sh)

echo "Submitted sample array job: $array_jobid"

# submit merge job, dependent on the whole array finishing successfully
sbatch --dependency=afterok:${array_jobid} \
       --export=ALL,NUM_SAMPLES=${NUM_SAMPLES},NUM_MEAS=${NUM_MEAS} \
       experiments/slurm/data_gen/merge.sh

# sbatch --export=ALL,NUM_SAMPLES=${NUM_SAMPLES},NUM_MEAS=${NUM_MEAS} \
#        experiments/slurm/data_gen/merge.sh
