#!/bin/bash

set -e

DATADIR="$1"
OUTDIR="$2"
state="$3"
version="$4"
mode="$5"
num_runs="$6"
num_data_points="$7"
sample_factor="$8"
PT="$9"
min_qb="${10}"
max_qb="${11}"
order="${12}"
log2_num_thread="${13}"
use_simd="${14}"
var_offset="${15}"
run_julia="${16}"
run_QFI="${17}"
use_perf="${18}"

# -------------------------
# check validity of input pars
# -------------------------

is_int() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_float() {
    [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]]
}

die() {
    echo "Error: $1" >&2
    exit 1
}

# directory checks
[[ -d "${DATADIR}" ]] || die "state directory does not exist: ${DATADIR}"

if [[ -z "$(ls -A "${DATADIR}" 2>/dev/null)" ]]; then
    die "state directory is empty: ${DATADIR}"
fi

# categorical checks
[[ "$version" == running* || "$version" == batched* ]] || die "invalid version: $version"

[[ "$mode" == "bench" || "$mode" == "test" ]] || die "invalid mode: $mode"

# numeric checks
is_uint "$num_runs" || die "num_runs must be int > 0"
(( num_runs > 0 )) || die "num_runs must be > 0"

is_uint "$num_data_points" || die "num_data_points must be int >= 0"
(( num_data_points >= 0 )) || die "num_data_points must be > 0"

is_float "$sample_factor" || die "sample_factor must be float"
awk "BEGIN {exit !($sample_factor > 0 && $sample_factor <= 1)}" || \
    die "sample_factor must be in (0, 1]"

# est / est_pt
is_uint "$PT" || die "PT must be 0 or 1"
[[ "$PT" == "0" || "$PT" == "1" ]] || die "PT must be 0 or 1"

[[ "$state" == "mm" && "$PT" -eq 1 ]] && die "mm does not support PT"

# min_qb, max_qb, order
is_uint "$min_qb" || die "min_qb must be int: ${min_qb}"
(( min_qb > 1 )) || die "min_qb must be > 1"

is_uint "$max_qb" || die "max_qb must be int: ${max_qb}"
(( max_qb > 1 )) || die "max_qb must be > 1"

(( min_qb <= max_qb )) || die "min_qb must be <= max_qb"

is_uint "$order" || die "order must be int"
(( order > 1 )) || die "order must be > 1"

# log2 threads: -1,0,1,2,...
is_int "$log2_num_thread" || die "log2_num_thread must be int"
(( log2_num_thread >= -1 )) || die "log2_num_thread must be >= -1"

# use_simd
[[ "$use_simd" == "0" || "$use_simd" == "1" ]] || die "use_simd must be 0 or 1"

# var_offset > 10
is_uint "$var_offset" || die "var_offset must be int >= 0"
(( var_offset > 10 )) || die "var_offset must be > 10"

# -------------------------
# done
# -------------------------


# experiment name (must be name of this dir and of runnable .cpp file)
exp_dir="experiments"
main_dir="main"
scripts_dir="scripts"
binary_name="bench_moments"
if (( run_QFI == 1 )); then
    binary_name="bench_QFI"    
fi
julia_script="${exp_dir}/${scripts_dir}/RM_batched.jl"

# --- dir management ---
BINDIR="build/${mode}/${exp_dir}/${main_dir}" # dir where binary lives

DATAOUTDIR="${OUTDIR}/data"

if (( run_julia == 1 )); then
    DATAOUTDIR="${DATAOUTDIR}_jl"
fi

mkdir -p "${OUTDIR}" # create OUTDIR if !exist
rm -rf "${DATAOUTDIR}"
mkdir -p "${DATAOUTDIR}" # create DATAOUTDIR if !exist

# --- bin and file management ---
BIN="${BINDIR}/${binary_name}" # binary
LOGFILE="${OUTDIR}/${version}.log" # log file (cerr)
OUTFILE="${OUTDIR}/${version}.txt" # out file (cout)
> "${LOGFILE}" # reset
> "${OUTFILE}" # reset

FREQFILE="${DATAOUTDIR}/frequencies"
echo "freq" > "${FREQFILE}"

for DATAINFILE in "${DATADIR}"/*.npz; do

    if [ ! -f "${DATAINFILE}" ]; then
        echo "File ${DATAINFILE} does not exist, exiting."
        exit 1
    fi

    # skip files with too many qubits
    filename=$(basename "$DATAINFILE")
    qb_str="${filename#qb_}"          # "qb_"  -> "02.npz"
    qb_str="${qb_str%.npz}"           # "02.npz" -> "02"
    qb_num=$((10#${qb_str}))          # force base-10 -> 2
    if (( qb_num > max_qb )); then
        continue
    fi        
    if (( qb_num < min_qb )); then
        continue
    fi         

    DATAOUTFILE="${DATAOUTDIR}/$(basename "${DATAINFILE}")"

    echo -e "Working on ${DATAINFILE} ...\n"
    echo -e "------------- START ${DATAINFILE} -------------\n" >> "${LOGFILE}"
    echo -e "------------- START ${DATAINFILE} -------------\n" >> "${OUTFILE}"

    if (( run_julia == 0 )); then

        PERFTMP=$(mktemp)
    
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            LAUNCH=(srun --mpi=pmix_v5  --cpu-bind=none perf stat -e cycles,task-clock -- ./"${BIN}") # slurm run
        elif (( use_perf == 1 )); then
            LAUNCH=(perf stat -e cycles,task-clock -- ./"${BIN}") # local run w/o slurm
        else
            echo "perf not used. We assume base frequency."
            LAUNCH=(./"${BIN}") # local run w/o slurm
        fi

        "${LAUNCH[@]}" \
            "${DATAINFILE}" \
            "${DATAOUTFILE}" \
            "${version}" \
            "${num_runs}" \
            "${num_data_points}" \
            "${sample_factor}" \
            "${PT}" \
            "${order}" \
            "${log2_num_thread}" \
            "${use_simd}" \
            "${var_offset}" \
            >>"${OUTFILE}" 2> >(tee -a "${LOGFILE}" | tee "${PERFTMP}" >&2) # stdout >> OUTFILE, stderr > console and >> LOGFILE

        FREQ_GHZ=$(grep -oP '\d+\.\d+(?=\s*GHz)' "${PERFTMP}" | tail -1)
        echo "${FREQ_GHZ}" >> "${FREQFILE}"
        rm -f "${PERFTMP}"
    fi

    if (( run_julia == 1 )); then  
        julia --project=julia -t "${order}" "${julia_script}" \
            --input "${DATAINFILE}" \
            --output "${DATAOUTFILE}" \
            --m "${order}" 
    fi        

    echo -e "------------- END ${DATAINFILE} -------------\n" >> "${LOGFILE}"
    echo -e "------------- END ${DATAINFILE} -------------\n" >> "${OUTFILE}"
done



