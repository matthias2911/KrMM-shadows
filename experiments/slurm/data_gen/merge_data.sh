#!/bin/bash
set -e

EXP_DIR="experiments"
py_merge_script="${EXP_DIR}/scripts/data_gen/merge_shadows_data.py"

site="$1"
num_samples="$2"
num_meas="$3"

case "$site" in
hpcgpu2|hpcgpu3) gen_data_path="/data/mgruber/shadows-cpp/${EXP_DIR}" ;;
vsc5)            gen_data_path="/gpfs/data/fs73274/mgruber/shadows-cpp/${EXP_DIR}" ;;
*) echo "Usage: $0 {hpcgpu2|hpcgpu3|vsc5} <num_samples> <num_meas>"; exit 1 ;;
esac

state="werner"
name="${state}_0.95_5"
werner_par="0.95"
order=128
data_path="${gen_data_path}/${name}"

python "${py_merge_script}" "${data_path}"

total_num_meas=$((num_samples * num_meas))
echo "state = ${state}" > "${data_path}/info"
echo "werner_par = ${werner_par}" >> "${data_path}/info"
echo "num_meas = ${total_num_meas}" >> "${data_path}/info"
echo "order = ${order}" >> "${data_path}/info"

echo "EXIT_SUCCESS"