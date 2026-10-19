#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np


def print_npz(path, data):
    print(f"\n--- {path} ---")
    for key in data.files:
        value = data[key]
        print(f"{key}:")
        print(f"  shape={value.shape}, dtype={value.dtype}")
        print(f"  value={value}")


def merge_files(target_dir: Path):
    sample_dirs = sorted(target_dir.glob("sample_*"))

    assert sample_dirs, f"No sample_* directories found in {target_dir}"

    # Collect files by filename.
    files_by_name = {}

    for sample_dir in sample_dirs:
        for path in sample_dir.glob("qb_*.npz"):
            files_by_name.setdefault(path.name, []).append(path)

    assert files_by_name, f"No qb_*.npz files found in {target_dir}"

    # These fields must be identical in every input file.
    fixed_fields = {
        "num_qubits",
        "max_order",
        "pt",
        "werner",
        "werner_par",
        "transposes",
        "moments",
        "moments_pt",
    }

    variable_fields = {
        "num_meas",
        "inds",
        "clicks",
    }

    expected_fields = fixed_fields | variable_fields

    for filename, files in sorted(files_by_name.items()):
        print(f"\n{'=' * 80}")
        print(f"Merging {filename}: {len(files)} files")
        print(f"{'=' * 80}")

        # Load and print the first file.
        reference = np.load(files[0], allow_pickle=False)
        # print_npz(files[0], reference)

        assert set(reference.files) == expected_fields, (
            f"{files[0]} has unexpected fields:\n"
            f"  found:    {set(reference.files)}\n"
            f"  expected: {expected_fields}"
        )

        total_num_meas = 0
        all_inds = []
        all_clicks = []

        for path in files:
            data = np.load(path, allow_pickle=False)

            # Print every file for debugging.
            # print_npz(path, data)

            # Check that the fields are exactly as expected.
            assert set(data.files) == expected_fields, (
                f"{path} has unexpected fields:\n"
                f"  found:    {set(data.files)}\n"
                f"  expected: {expected_fields}"
            )

            # Check all fields that must be identical.
            for key in fixed_fields:
                np.testing.assert_array_equal(
                    data[key],
                    reference[key],
                    err_msg=f"{path}: '{key}' differs from {files[0]}",
                )

            # num_meas must agree with the number of measurements.
            num_meas = int(data["num_meas"][0])

            total_num_meas += num_meas

            all_inds.append(data["inds"])
            all_clicks.append(data["clicks"])

        # Concatenate measurements from all samples.
        merged_inds = np.concatenate(all_inds)
        merged_clicks = np.concatenate(all_clicks)

        # Construct the merged file.
        merged = {
            key: reference[key]
            for key in fixed_fields
        }

        merged["num_meas"] = np.array(
            [total_num_meas],
            dtype=np.int32,
        )

        merged["inds"] = merged_inds
        merged["clicks"] = merged_clicks

        output_path = target_dir / filename

        np.savez(output_path, **merged)

        # Load the resulting file and print it as well.
        final_data = np.load(output_path, allow_pickle=False)
        # print_npz(output_path, final_data)

        print(
            f"\nSuccessfully merged {filename}: "
            f"{len(files)} files, "
            f"{total_num_meas} measurements"
        )

    # Only delete sample directories after ALL merges succeeded.
    print(f"\n{'=' * 80}")
    print("Removing sample directories")
    print(f"{'=' * 80}")

    for sample_dir in sample_dirs:
        print(f"Removing {sample_dir}")
        shutil.rmtree(sample_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Merge qb_*.npz files from sample_* directories."
    )

    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory containing sample_* subdirectories",
    )

    args = parser.parse_args()

    merge_files(args.target_dir)


if __name__ == "__main__":
    main()