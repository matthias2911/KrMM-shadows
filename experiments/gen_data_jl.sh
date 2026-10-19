#!/bin/bash

# set -e

# file management
EXP_DIR="experiments"
py_sim_script="${EXP_DIR}/scripts/data_gen/sim_shadows_data_jl.py"

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
num_qubits=(2 4 6 8 10)
# NMs=(326797 36310 4034 448 49) # 1e8
# NMs=(32679 3631 403 44 4) # 1e7
NMs=(3267 363 40 4 1) # 1e6

order=34
werner_pars="0.95"
name="werner_0.95_julia_1e8"

data_path="${gen_data_path}/${name}"

for werner_par in $werner_pars; do
    for i in "${!num_qubits[@]}"; do
        nqb=${num_qubits[$i]}
        NM=${NMs[$i]}

        start_qb=$SECONDS
        python $py_sim_script -qb ${nqb} -m ${NM} -o ${order} -t ${werner_par} -p ${data_path} #--random_settings 1
        echo "time ${nqb} qubits: $((SECONDS - start_qb))s"
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
