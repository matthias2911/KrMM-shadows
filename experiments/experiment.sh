#!/bin/bash

set -e

start=$SECONDS

die() {
    echo "Error: $1" >&2
    exit 1
}

exp_dir="experiments"
CFG="${exp_dir}/experiments.json" # exp config file
hw_file="${exp_dir}/hardware.json" # hardware file

# experiment name
EXP="$1"
jq -e ".${EXP}" "$CFG" >/dev/null 2>&1 || die "experiment '${EXP}' not found in $CFG"

# read in experiment pars from json file
run_exp=$(jq -r ".${EXP}.run_exp" "$CFG") # 0 or 1
rm_old_exp=$(jq -r ".${EXP}.rm_old_exp" "$CFG") # 0 or 1
single_plots=$(jq -r ".${EXP}.single_plots" "$CFG") # 0 or 1
run_QFI=$(jq -r ".${EXP}.run_QFI // 0" "$CFG") # 0 or 1
run_julia=$(jq -r ".${EXP}.run_julia // 0" "$CFG") # 0 or 1
manual_cmp=$(jq -r ".${EXP}.manual_cmp // 0" "$CFG") # 0 or 1
[[ "$run_exp" == "0" || "$run_exp" == "1" ]] || die "run_exp must be 0 or 1"
[[ "$single_plots" == "0" || "$single_plots" == "1" ]] || die "single_plots must be 0 or 1"
[[ "$run_julia" == "0" || "$run_julia" == "1" ]] || die "run_julia must be 0 or 1"
[[ "$manual_cmp" == "0" || "$manual_cmp" == "1" ]] || die "manual_cmp must be 0 or 1"

state=$(jq -r ".${EXP}.state" "$CFG") # bench or test
mapfile -t versions < <(jq -r ".${EXP}.versions[]" "$CFG") # running or batched
mode=$(jq -r ".${EXP}.mode" "$CFG") # bench or test

num_runs=$(jq -r ".${EXP}.num_runs" "$CFG") # > 0
num_data_points=$(jq -r ".${EXP}.num_data_points" "$CFG") # >= 0
mapfile -t sample_factors < <(jq -r ".${EXP}.sample_factors[]" "$CFG")
PT=$(jq -r ".${EXP}.PT" "$CFG") # 0 or 1

min_qb=$(jq -r ".${EXP}.min_qb" "$CFG") # > 1
max_qb=$(jq -r ".${EXP}.max_qb" "$CFG") # > 1
order=$(jq -r ".${EXP}.order" "$CFG") # 1 < x <= max_order (data dependent)
tol=$(jq -r ".${EXP}.tol" "$CFG") # 0 < x <= 1
CI_perc=$(jq -r ".${EXP}.CI_perc" "$CFG") # 0 < x <= 1

mapfile -t log2_num_threads < <(jq -r ".${EXP}.log2_num_threads[]" "$CFG") # -1, 0, 1, 2, ...
mapfile -t use_simds < <(jq -r ".${EXP}.use_simds[]" "$CFG") # 0 or 1
var_offset=$(jq -r ".${EXP}.var_offset" "$CFG") # < num_meas / 2

data_in_dir=$(jq -r ".${EXP}.data_in_dir" "$CFG") # bench or test
hw=$(jq -r ".${EXP}.hw" "$CFG") # bench or test
use_perf=$(jq -r ".${EXP}.use_perf // 0" "$CFG") # 0 or 1
mapfile -t plot_scripts < <(jq -r ".${EXP}.plot_scripts[]" "$CFG") # pure filename

n=${#sample_factors[@]}
[[ ${#log2_num_threads[@]}   -eq $n ]] || { echo "Length mismatch: log2_num_threads" >&2; exit 1; }
[[ ${#use_simds[@]}          -eq $n ]] || { echo "Length mismatch: use_simds" >&2; exit 1; }
[[ ${#versions[@]}           -eq $n ]] || { echo "Length mismatch: versions" >&2; exit 1; }

# hardcoded paths
scripts_dir="scripts"
plotter_dir="plotting"
results_dir="results"
runner="${exp_dir}/${scripts_dir}/run.sh"
single_plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/plot.py"
if (( run_QFI == 1 )); then
    single_plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/plot_QFI.py"    
fi
cmp_plotter_jl="${exp_dir}/${scripts_dir}/${plotter_dir}/cmp_to_bs.py"
manual_cmp_plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/manual_cmp.py"

# configure and build project
./scripts/configure.sh
./scripts/build.sh

root_exp_dir="${exp_dir}/${results_dir}/${EXP}"
if (( run_exp == 1 )); then
    if [[ -d "$root_exp_dir" && "$rm_old_exp" -eq 1 ]]; then
        old_exp_dir="${exp_dir}/${results_dir}/old"
        root_old_exp_dir="${exp_dir}/${results_dir}/old/${EXP}"
        mkdir -p  "${old_exp_dir}"
        rm -rf "${root_old_exp_dir}"
        mv "${root_exp_dir}" "${old_exp_dir}"
    fi
fi

for ((i = 0; i < n; i++)); do
    sample_factor="${sample_factors[i]}"
    log2_num_thread="${log2_num_threads[i]}"
    use_simd="${use_simds[i]}"
    version="${versions[i]}"

    DATADIR="${data_in_dir}/${state}"
    OUTDIR="${root_exp_dir}/${state}/${version}"

    if (( run_exp == 1 )); then
        echo -e "Running ${OUTDIR} ...\n"
        ./"${runner}" \
            "${DATADIR}" \
            "${OUTDIR}" \
            "${state}" \
            "${version}" \
            "${mode}" \
            "${num_runs}" \
            "${num_data_points}" \
            "${sample_factor}" \
            "${PT}" \
            "${min_qb}" \
            "${max_qb}" \
            "${order}" \
            "${log2_num_thread}" \
            "${use_simd}" \
            "${var_offset}" \
            "${run_julia}" \
            "${run_QFI}" \
            "${use_perf}"
        echo -e "Done\n"
    fi

    if (( single_plots == 1 )); then

        echo -e "Plotting ${OUTDIR} ...\n"

        DATAOUTDIR="${OUTDIR}/data"
        PLOTOUTDIR="${OUTDIR}/plots"

        if (( run_julia == 1 )); then  
            DATAOUTDIR="${DATAOUTDIR}_jl"
            PLOTOUTDIR="${PLOTOUTDIR}_jl"                            
        fi

        [[ -d "${DATAOUTDIR}" ]] || die "result data directory does not exist: ${DATAOUTDIR}"
        mkdir -p "${PLOTOUTDIR}"                  

        python "${single_plotter}" \
            --data_dir "${DATAOUTDIR}" \
            --out_dir "${PLOTOUTDIR}" \
            --state "${state}" \
            --CI_perc "${CI_perc}" \
            --hw_file "${hw_file}" \
            --hardware "${hw}" \
            --tol "${tol}" \
            --julia "${run_julia}" \
            --version "${version}" 

        echo -e "Done\n"
    fi                

done

if (( run_julia == 0 )); then  
    for plot_script in "${plot_scripts[@]}"; do
        plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/${plot_script}"

        ./"${plotter}" \
            "${root_exp_dir}" \
            "${tol}" \
            "${CI_perc}" \
            "${hw}" \
            "${run_QFI}" 
        
    done
fi

if (( manual_cmp == 1 )); then  

    datadirs=()
    OUTDIR="${root_exp_dir}/${state}/manual_cmp/plots"

    mkdir -p "${OUTDIR}"                  

    for ((i = 0; i < n; i++)); do
        sample_factor="${sample_factors[i]}"
        log2_num_thread="${log2_num_threads[i]}"
        use_simd="${use_simds[i]}"
        version="${versions[i]}"

        DATADIR="${root_exp_dir}/${state}/${version}/data"
        datadirs+=("$DATADIR")
    done

    python "${manual_cmp_plotter}" \
            --data_dirs "${datadirs[@]}" \
            --out_dir "${OUTDIR}" \
            --CI_perc "${CI_perc}" \
            --hw_file "${hw_file}" \
            --hardware "${hw}" \
            --tol "${tol}"            

fi

if (( run_julia == 1 )); then  

    for ((i = 0; i < n; i++)); do
        sample_factor="${sample_factors[i]}"
        log2_num_thread="${log2_num_threads[i]}"
        use_simd="${use_simds[i]}"
        version="${versions[i]}"

        DATADIR="${root_exp_dir}/${state}/${version}/data"
        DATADIR_jl="${root_exp_dir}/${state}/${version}/data_jl"
        PLOT_DIR="${root_exp_dir}/${state}/${version}/cmp_to_bs"

        [[ -d "${DATADIR}" ]] || die "result data directory does not exist: ${DATADIR}"
        [[ -d "${DATADIR_jl}" ]] || die "result data directory does not exist: ${DATADIR_jl}"
        mkdir -p "${PLOT_DIR}"                                    

        python "${cmp_plotter_jl}" \
            --data_dir1 "${DATADIR}" \
            --data_dir2 "${DATADIR_jl}" \
            --out_dir "${PLOT_DIR}" \
            --CI_perc "${CI_perc}" \
            --hw_file "${hw_file}" \
            --hardware "${hw}" \
            --tol "${tol}"             

    done   

fi

elapsed=$((SECONDS - start))
echo "Wall time: ${elapsed}s"
