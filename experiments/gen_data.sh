#!/bin/bash

# set -e

# file management
EXP_DIR="experiments"
py_sim_script="${EXP_DIR}/scripts/data_gen/sim_shadows_data.py"
py_merge_script="${EXP_DIR}/scripts/data_gen/merge_shadows_data.py"

case "$1" in
    hpcgpu2)
        gen_data_path="/data/mgruber/shadows-cpp/${EXP_DIR}"
        ;;
    hpcgpu3)
        gen_data_path="/home/mgruber/data/${EXP_DIR}"
        ;;        
    vsc5)
        gen_data_path="/gpfs/data/fs73274/mgruber/shadows-cpp/${EXP_DIR}"
        ;;          
    test)
        gen_data_path="data"
        ;;                 
    *)
        echo "Usage: $0 {hpcgpu2|hpcgpu3|vsc5}"
        exit 1
        ;;
esac

echo "Using: $gen_data_path"

start=$SECONDS

# # -------- ghz state --------- #
# num_qubits="2 3 4 5 6 7 8 9 10" # should take about 12h
# num_meas=110000000
# order=128
# state="ghz"
# name="ghz_14qb_10000"

# # local pars
# data_path="${gen_data_path}/${name}"
# # rm -rf "${data_path}"
# for nqb in $num_qubits; do
#     python $py_sim_script -qb ${nqb} --shots ${num_meas} --order ${order} -s ${state} --path ${data_path}
# done

# echo "state = ${state}" > "${data_path}/info"
# echo "num_meas = ${num_meas}" >> "${data_path}/info"
# echo "order = ${order}" >> "${data_path}/info"
# # -------- ghz state --------- #

# -------- werner state --------- #
# num_samples=1
# compute_probs=1
# num_qubits="2 4"
# werner_pars="0.95"
# num_meas=1000
# order=128
# state="werner"
# ranks="1"
# name="${state}_0.95_test_test"

# -------- maximally mixed state --------- #
num_samples=1
compute_probs=1
num_qubits="2 3 4"
werner_pars="0.95"
num_meas=10000000
order=16
state="mm"
ranks="1"
name="${state}_test"

# -------- random rank-4 state --------- #
# num_samples=1
# compute_probs=1
# num_qubits="3 4 5 6 7 8"
# werner_pars="0.0"
# num_meas=110000000
# order=10
# state="rankX"
# ranks="2 4 6"
# name="rank"

# local pars
for werner_par in $werner_pars; do
    for rank in $ranks; do

        if (( num_samples > 1 )); then
            for ((run=0; run<num_samples; run++)); do
                data_path="${gen_data_path}/${name}"
                sample_path="${data_path}/sample_${run}"

                # rm -rf "${data_path}"

                for nqb in $num_qubits; do
                    start_qb=$SECONDS
                    python $py_sim_script -qb ${nqb} --shots ${num_meas} --order ${order} -s ${state} -t ${werner_par} --path ${sample_path} --rank ${rank}
                    elapsed_qb=$((SECONDS - start_qb))
                    echo "time ${nqb} qubits: ${elapsed_qb}s"                  
                done
            done

            python "${py_merge_script}" "${data_path}"

            total_num_meas=$((num_samples * num_meas))
            echo "state = ${state}" > "${data_path}/info"
            echo "werner_par = ${werner_par}" >> "${data_path}/info"
            echo "num_meas = ${total_num_meas}" >> "${data_path}/info"
            echo "order = ${order}" >> "${data_path}/info"        
        else
            data_path="${gen_data_path}/${name}"

            # rm -rf "${data_path}"

            for nqb in $num_qubits; do
                start_qb=$SECONDS
                python $py_sim_script -qb ${nqb} --shots ${num_meas} --order ${order} -s ${state} -t ${werner_par} --path ${data_path} --compute_probs ${compute_probs} --rank ${rank}
                elapsed_qb=$((SECONDS - start_qb))
                echo "time ${nqb} qubits: ${elapsed_qb}s"            
            done

            echo "state = ${state}" > "${data_path}/info"
            echo "werner_par = ${werner_par}" >> "${data_path}/info"
            echo "num_meas = ${num_meas}" >> "${data_path}/info"
            echo "order = ${order}" >> "${data_path}/info"
        fi

    done
done
# -------- werner state --------- #

# # -------- maximally mixed state --------- #
# num_qubits="5"
# num_meas=1000000000
# order=256
# state="mm"
# name="mm_5qb"

# # local pars
# data_path="${gen_data_path}/${name}"
# # rm -rf "${data_path}"
# for nqb in $num_qubits; do
#     python $py_sim_script -qb ${nqb} --shots ${num_meas} --order ${order} -s ${state} --path ${data_path}
# done

# echo "state = ${state}" > "${data_path}/info"
# echo "num_meas = ${num_meas}" >> "${data_path}/info"
# echo "order = ${order}" >> "${data_path}/info"
# # -------- maximally mixed state --------- #


elapsed=$((SECONDS - start))
echo "Wall time: ${elapsed}s"


echo "EXIT_SUCCESS"
