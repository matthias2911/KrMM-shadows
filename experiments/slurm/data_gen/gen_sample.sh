#!/bin/bash
set -e

EXP_DIR="experiments"
py_sim_script="${EXP_DIR}/scripts/data_gen/sim_shadows_data.py"

site="$1"
run="$2"

case "$site" in
hpcgpu2|hpcgpu3)
    gen_data_path="/data/mgruber/shadows-cpp/${EXP_DIR}"
    ;;
vsc5)
    gen_data_path="/gpfs/data/fs73274/mgruber/shadows-cpp/${EXP_DIR}"
    ;;
*)
    echo "Usage: $0 {hpcgpu2|hpcgpu3|vsc5} <run_index>"
    exit 1
    ;;
esac

num_qubits="10"
compute_probs=0
werner_par="0.95"
num_meas=25000000
order=128
state="werner"
name="${state}_0.95_5"


probs_name="${state}_0.95"

data_path="${gen_data_path}/${name}"
sample_path="${data_path}/sample_${run}"
probs_dir="${gen_data_path}/${probs_name}/probs"

for nqb in $num_qubits; do
    start_qb=$SECONDS
    python $py_sim_script -qb ${nqb} --shots ${num_meas} --order ${order} \
        -s ${state} -t ${werner_par} --path ${sample_path} --compute_probs ${compute_probs} \
        --probs_dir ${probs_dir}
    elapsed_qb=$((SECONDS - start_qb))
    echo "time ${nqb} qubits: ${elapsed_qb}s"
done

echo "EXIT_SUCCESS"