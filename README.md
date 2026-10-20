# KrMM-shadows

This is the repository accompanying the publication "**KrMM Recurrences: HPC-Enhanced Estimators for Nonlinear Classical Shadows**" ([arXiv](https://doi.org/10.48550/arXiv.2610.26472)).

## Project Structure

```
.
├── experiments/   # all experiments (data generation, running, plotting, etc.)
├── external/      # external C++ projects (third-party libraries)
├── include/       # headers for including the entire C++ project
├── julia/         # TOML files for setting up the Julia environment
├── scripts/       # scripts for building and configuring the project
└── src/           # C++ source code of this repository's project
```

The experimental data and the plotting scripts used to obtain all figures in the paper can be found under `experiments/combined_plots`.

## Requirements

- x86-64 UNIX-based system (not tested on ARM or other operating systems)
- zlib (preinstalled on most Linux systems)

## Citation

If you use this repository in your work, please cite our [arXiv paper](https://doi.org/10.48550/arXiv.2610.26472) (see also `CITATION.cff`).

## Quick Start: Small Test Experiment

This describes the full end-to-end pipeline for a small test experiment.

### 1. Configure hardware information

Add your hardware information under `"test_HW"` in `experiments/hardware.json`. For a minimal test, it is enough to enter the core count, frequency and vector size (in bytes).

Optionally, if you want to measure the CPU frequency, install `perf` and set `use_perf` to `1` in the `"test"` experiment inside `experiments/experiments.json`.

### 2. Install dependencies

Install the dependencies from `shadows.yml` and `shadows-dev.yml`. This can be done smoothly with a Conda-compatible package manager, e.g. micromamba. With micromamba:

```bash
micromamba create -n shadows_env -f shadows.yml
micromamba install -n shadows_env -f shadows-dev.yml
micromamba activate shadows_env
```

### 3. Generate test data

```bash
./experiments/gen_data.sh test
```

### 4. Run the experiments

This takes upwards of one minute, depending on your hardware.

```bash
./experiments/experiment.sh test
```

## Julia

To run the Julia baseline, first set up the Julia environment from the files in `julia/` as:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
