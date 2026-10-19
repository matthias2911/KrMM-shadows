import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import numpy as np
import sys
import os
import json

from roofline import Hardware
from plot import load_data, extract_data, log_run_details, compute_PP, lighten, compute_AI, compute_PP_vs_n, compute_BWs, plot_rooflines, scatter_roofline_data_points


####################################################################################
####################################################################################
    # PLOTTING SCRIPT FOR DEVELOPMENT ONLY
    # FINAL PLOTS WERE ALL OBTAINED WITH NOTEBOOKS IN "combined_plots"
    # SOME PLOTTING FUNCTIONS MAY BE OUTDATED
####################################################################################
####################################################################################


def cmp_RE_vs_order(ax, args, data1, data2, title=True):
    for idx, n in enumerate(data1["qubits"]):
        c = data1["colors"][idx%len(data1["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]        
        ax.semilogy(data1["orders"], data1["RE"][idx], c=c, marker=marker, label=f"n={n}, 1")
        ax.semilogy(data2["orders"], data2["RE"][idx], c=lighten(c), marker=marker, label=f"n={n}, 2")
    ax.set_xlabel("order")
    ax.set_ylabel("RE")
    if title:
        ax.set_title(f"median rel. error (RE) of 2 to 1\nof moment estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_frac_to_plot = 1e2
    max_frac = np.max(data2["RE"][idx]/data1["RE"][idx])
    if max_frac > max_frac_to_plot:
        ax.set_ylim(ymin=None, ymax=max_frac_to_plot)      

def plot_RE_vs_T_fixed_n(ax, args, data, datas_batched, title=True, n=2):
    assert n <= data["qubits"][-1]
    for data_batched in datas_batched:
        assert n <= data_batched["qubits"][-1]

    # -------------------- RUNNING -------------------- #
    nidx = np.where(data["qubits"] == n)[0][0]
    plt_orders = [2, 8, 16]
    iidx = 0
    for idx, order in enumerate(data["orders"]):
        # if (order & (order - 1)) != 0: # pow of 2
        #     continue
        if order not in plt_orders:
            continue
        iidx += 1
        midx = order - 2
        c = data["colors"][iidx%len(data["colors"])]
        marker = data["markers"][iidx%len(data["markers"])] if data["num_data_points"] < 30 else None      
        ax.loglog(data["inter_tidx"][nidx] + midx, data["inter_RE"][nidx, midx], lw=1, c=c, marker=marker, linestyle="dotted")    
    iidx = 0
    for idx, order in enumerate(data["orders"]):
        # if (order & (order - 1)) != 0: # pow of 2
        #     continue
        if order not in plt_orders:
            continue
        iidx += 1
        midx = order - 2
        c = data["colors"][iidx%len(data["colors"])]
        marker = data["markers"][iidx%len(data["markers"])] if data["num_data_points"] < 30 else None      
        ax.loglog(data["inter_tidx"][nidx] + midx, data["inter_ERE"][nidx, midx], c=c, marker=marker, label=f"k={order}", lw=2.5)

    # -------------------- BATCHED -------------------- #
    for data_batched in datas_batched:
        nidx = np.where(data_batched["qubits"] == n)[0][0]
        iidx = 0
        for idx, order in enumerate(data_batched["orders"]):
            if order not in plt_orders:
                continue
            iidx += 1
            midx = order - 2
            c = data_batched["colors"][iidx%len(data_batched["colors"])]
            marker = data_batched["markers"][iidx%len(data_batched["markers"])]
            ax.loglog(data_batched["num_meas"], data_batched["RE"][nidx, midx], linestyle="none", c=c, marker=marker, ms=7)            

    ax.set_xlabel(r"$N_U$")
    ax.set_ylabel("relative error")
    if title:
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof n={n} moment estimation over #shots for different k\n median over {data["num_runs"]} runs")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    manual = False
    if manual:
        min_RE_to_plot = 1e-4
        max_RE_to_plot = 3e1
        min_T = 2e5

        min_RE = np.min(data["inter_RE"][nidx, :])
        max_RE = np.max(data["inter_RE"][nidx, :])
        # min_RE = min_RE_to_plot if min_RE < min_RE_to_plot else None
        min_RE = min_RE / 2 if not np.isnan(min_RE) else min_RE_to_plot
        max_RE = max_RE_to_plot if max_RE > max_RE_to_plot or np.isnan(max_RE) else max_RE * 2
        ax.set_ylim(ymin=1e-3, ymax=max_RE)      
        ax.set_xlim(xmin=min_T, xmax=None)      
    else:
        min_RE_to_plot = 1e-4
        max_RE_to_plot = 1e3
        min_T = 1e3

        min_RE = np.min(data["inter_RE"][nidx, :])
        max_RE = np.max(data["inter_RE"][nidx, :])
        # min_RE = min_RE_to_plot if min_RE < min_RE_to_plot else None
        min_RE = min_RE / 2 if not np.isnan(min_RE) else min_RE_to_plot
        max_RE = max_RE_to_plot if max_RE > max_RE_to_plot or np.isnan(max_RE) else max_RE * 2
        ax.set_ylim(ymin=min_RE, ymax=max_RE)      
        ax.set_xlim(xmin=min_T, xmax=None)           





# ---------------------- main ----------------------
def run_all_plots(args):

    data_dirs = args.data_dirs

    data_running = 0
    datas_batched = []

    # extract data
    runnings = 0
    for data_dir in data_dirs:
        in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points = load_data(args, data_dir)
        data = extract_data(args, in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points, compute_DICI=False)

        data["colors"] = [x for x in mcolors.TABLEAU_COLORS.keys()]
        data["markers"] = ["o", "v", "p", "*", "s", "X"]
        data["linestyles"] = ["-", ":", "-."]
        data["lw"] = 1

        if "batched" in data_dir:
            datas_batched.append(data)
        elif "running" in data_dir:         
            data_running = data
            runnings += 1
        else:
            sys.exit(f"No correct version")

    if runnings != 1:
        sys.exit(f"Incorrect number of running versions: {runnings}")

    # dirs
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(single_plt_dir, exist_ok=True)    

    # ----------------------- plots ----------------------- #

    for n in data_running["qubits"]:
        fg, ax = plt.subplots(layout="constrained")
        plot_RE_vs_T_fixed_n(ax, args, data_running, datas_batched, title=False, n=n)

        trg_dir = f"{single_plt_dir}/RE_vs_T_fixed_n"
        os.makedirs(trg_dir, exist_ok=True)    
        fg.savefig(f"{trg_dir}/n_{n}.png")
        fg.savefig(f"{trg_dir}/n_{n}.pdf", bbox_inches="tight")          
        plt.close(fg) 

    print("\n----------------------------------------------------------------")
    print(f"Convergence overview can be found in {args.out_dir}.")
    print(f"More detailed pots can be found in the other dirs.")
    print(f"\nWARNING:\nThese are development plotting scripts.\nAll final plots were created with notebooks in experiments/combined_plots.")
    print("----------------------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare different moments runs against each other.')

    parser.add_argument(
        '--data_dirs',
        type=str,
        nargs='+',
        help='dirs to parse for data'
    )

    parser.add_argument('--out_dir', type=str, help='dir to store plots in')
    parser.add_argument('--CI_perc', type=int, help='percentage to use for conf. interval', default=99, choices=[95, 99])
    parser.add_argument('--hw_file', type=str, help='which hardware file to use')
    parser.add_argument('--hardware', type=str, help='which hardware was used for benchmarks', default="9354P")
    parser.add_argument('--tol', type=float, help='error tolerance for declaring convergence', default=0.1)

    args = parser.parse_args()

    for data_dir in args.data_dirs:
        if not os.path.isdir(data_dir):
            sys.exit(f"Error: {data_dir} does not exist or is not a directory")

    if not os.path.isdir(args.out_dir):
        sys.exit(f"Error: out_dir does not exist or is not a directory: {args.out_dir}")

    assert 0 < args.tol and args.tol <= 1

    # add hardcoded CI factor based on CI_perc
    if args.CI_perc == 95:
        args.CI_fac = 1.960
    elif args.CI_perc == 99:
        args.CI_fac = 2.576     

    # add hardcoded float size
    args.floatsize = 8 # B
    
    print(f"Plotting...")

    run_all_plots(args)
