#!/bin/bash

# compares all data dirs inside root_die against each other
# by calling cmp.py

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <root_dir>" >&2
    exit 1
fi

exp_dir="experiments"
scripts_dir="scripts"
plotter_dir="plotting"
plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/cmp.py"
hw_file="${exp_dir}/hardware.json"

ROOT_DIR="${1%/}"   # strip trailing slash
tol="$2"   # strip trailing slash
CI_perc="$3"   # strip trailing slash
hw="$4"   # strip trailing slash
run_QFI="$5"   # strip trailing slash

if (( run_QFI == 1 )); then
    plotter="${exp_dir}/${scripts_dir}/${plotter_dir}/cmp_QFI.py"
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Error: root dir '$ROOT_DIR' does not exist." >&2
    exit 1
fi

# extract all data dirs
mapfile -t DATA_DIRS < <(
    find "$ROOT_DIR" \
        -type d \
        -name "data" \
        -print \
        | sort
)

N="${#DATA_DIRS[@]}"
if [[ $N -eq 0 ]]; then
    echo "No 'data' directories found under '$ROOT_DIR'." >&2
    exit 1
fi

echo "Found $N data directories. Running $((N * (N - 1))) pairwise comparisons."
echo ""

# convert an absolute data-dir path to its relative counterpart
rel_path() {
    local abs="$1"
    local rel="${abs#"${ROOT_DIR}"/}" # remove ROOT_DIR prefix
    rel="${rel%/data}" # remove trailing /data
    echo "$rel"
}

# go through all pairs (i, j), i != j
SUCCESS=0
FAIL=0
SKIP=0

for ((i = 0; i < N; i++)); do
    DIR_A="${DATA_DIRS[$i]}"
    OUTBASE_A="${DIR_A%/data}" # parent of data/ is the version dir
    REL_A="$(rel_path "$DIR_A")"

    for ((j = 0; j < N; j++)); do
        [[ $i -eq $j ]] && continue

        DIR_B="${DATA_DIRS[$j]}"
        REL_B="$(rel_path "$DIR_B")"
        
        PLOT_DIR="${OUTBASE_A}/plots/cmp/${REL_B}" # build plot output dir

        echo "──────────────────────────────────────────────────────────"
        echo "  1 : $REL_A"
        echo "  2 : $REL_B"
        echo "  -> : ${PLOT_DIR#"${ROOT_DIR}/"}"

        mkdir -p "$PLOT_DIR"

        if python "${plotter}" \
            --data_dir1 "${DIR_A}" \
            --data_dir2 "${DIR_B}" \
            --out_dir "${PLOT_DIR}" \
            --CI_perc "${CI_perc}" \
            --hw_file "${hw_file}" \
            --hardware "${hw}" \
            --tol "${tol}"; then
            ((SUCCESS++)) || true
        else
            echo "  [WARNING] ${plotter} returned non-zero for the pair above" >&2
            ((FAIL++)) || true
        fi
    done
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done. Success: $SUCCESS  |  Failed: $FAIL"
