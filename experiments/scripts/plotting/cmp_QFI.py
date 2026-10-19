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
from plot_QFI import load_data, extract_data, log_run_details
from plot import compute_PP, lighten, compute_PP_vs_n  


####################################################################################
####################################################################################
    # PLOTTING SCRIPT FOR DEVELOPMENT ONLY
    # FINAL PLOTS WERE ALL OBTAINED WITH NOTEBOOKS IN "combined_plots"
    # SOME PLOTTING FUNCTIONS MAY BE OUTDATED
####################################################################################
####################################################################################


def cmp_RE_frac_vs_order(ax, args, data1, data2, title=True):

    for idx, n in enumerate(data1["qubits"]):
        c = data1["colors"][idx%len(data1["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])] if data1["num_data_points"] < 30 else None      
        ax.loglog(data1["inter_tidx"][idx], data2["inter_RE"][idx]/data1["inter_RE"][idx], c=c, marker=marker, label=f"n={n}")

    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75)
    ax.set_xlabel("order")
    ax.set_ylabel("RE2/RE1")
    if title:
        ax.set_title(f"Fraction of median rel. error (RE) of 2 to 1\nof moment estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    # max_frac_to_plot = 1e2
    # max_frac = np.max(data2["RE"][idx]/data1["RE"][idx])
    # if max_frac > max_frac_to_plot:
    #     ax.set_ylim(ymin=None, ymax=max_frac_to_plot)       

def cmp_SU_vs_n(ax, args, data1, data2, title=True):
    ax.plot(data1["qubits"], data2["runtimes"]/data1["runtimes"] , marker="v", label="speedup")
    ax.axhline(y=1, linestyle="--", c="slategrey", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("runtime2/runtime1")
    if title:
        ax.set_title(f"Speedup of 1 to 2 of median runtime for estimating all moments\nup to order {data1['orders'][-1]} over n")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def cmp_KrMM_SU_vs_n(ax, args, data1, data2, title=True):
    ax.plot(data1["qubits"], data2["runtimesDSC"]/data1["runtimesDSC"] , marker="v", label="speedup")
    ax.axhline(y=1, linestyle="--", c="slategrey", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("runtime2/runtime1")
    if title:
        ax.set_title(f"Speedup of 1 to 2 of median runtime for estimating all moments\nup to order {data1['orders'][-1]} over n")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def cmp_comp_HW_usage_vs_n(ax, args, data1, data2, title=True):
    cs1 = data1["colors"]
    cs2 = data2["colors"]

    ax.semilogy(data2["qubits"], data2["vector_width"], c=cs2[0], marker="o", label="vector-width, 2")
    ax.semilogy(data2["qubits"], data2["cores"], c=cs2[1], marker="s", label="#cores, 2")
    ax.semilogy(data2["qubits"], data2["num_threads"], c=cs2[2], marker="v", label="#threads, 2")

    ax.semilogy(data1["qubits"], data1["vector_width"], c=cs1[0], marker="o", label="vector-width, 1")
    ax.semilogy(data1["qubits"], data1["cores"], c=cs1[1], marker="s", label="#cores, 1")
    ax.semilogy(data1["qubits"], data1["num_threads"], c=cs1[2], marker="v", label="#threads, 1")

    ax.set_xlabel("n")
    ax.set_ylabel("#cores/ #threads/ #nodes/ vector-width [B]")
    if title:
        ax.set_title(f"Used number of threads, number of cores,\nand vector-widths in bytes over num_qubits")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

# ---------------------- main ----------------------
def run_all_plots(args):

    in_data1, qubits, n_star, orders, num_meas, num_runs, num_data_points = load_data(args, args.data_dir1)
    in_data2, qubits2, n_star2, orders2, num_meas2, num_runs2, num_data_points2 = load_data(args, args.data_dir2)

    assert np.all(qubits == qubits2)
    assert np.all(orders == orders2)
    assert n_star == n_star2
    assert num_runs == num_runs2
    assert num_data_points == num_data_points2

    data1 = extract_data(args, in_data1, qubits, n_star, orders, num_meas, num_runs, num_data_points, compute_DICI=False)
    data2 = extract_data(args, in_data2, qubits, n_star, orders, num_meas, num_runs, num_data_points, compute_DICI=False)

    # plot globals
    data1["colors"] = [x for x in mcolors.TABLEAU_COLORS.keys()]
    data1["markers"] = ["o", "v", "p", "*", "s", "X"]
    data1["linestyles"] = ["-", ":", "-."]
    data1["lw"] = 1

    data2["colors"] = [lighten(x) for x in mcolors.TABLEAU_COLORS.keys()]
    data2["markers"] = ["o", "v", "p", "*", "s", "X"]
    data2["linestyles"] = ["-", ":", "-."]
    data2["lw"] = 1

    # dirs
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(single_plt_dir, exist_ok=True)    


    # ----------------------- plots ----------------------- #

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_RE_frac_vs_order(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.png")
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.pdf", bbox_inches="tight")

    # --- rel. err over order --- #

    # --- SU over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_SU_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/SU_vs_n.png")
    fg.savefig(f"{single_plt_dir}/SU_vs_n.pdf", bbox_inches="tight")

    # --- SU over n --- #


    # --- krMM SU over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_KrMM_SU_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/KrMM_SU_vs_n.png")
    fg.savefig(f"{single_plt_dir}/KrMM_SU_vs_n.pdf", bbox_inches="tight")
    # --- krMM SU over n --- #    

    # --- comp. HW usage over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_comp_HW_usage_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/comp_HW_usage_vs_n.png")
    fg.savefig(f"{single_plt_dir}/comp_HW_usage_vs_n.pdf", bbox_inches="tight")

    # --- comp. HW usage over n --- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare different moments runs against each other.')
    parser.add_argument('--data_dir1', type=str, help='dir1 to parse for data')
    parser.add_argument('--data_dir2', type=str, help='dir2 to parse for data')
    parser.add_argument('--out_dir', type=str, help='dir to store plots in')
    parser.add_argument('--CI_perc', type=int, help='percentage to use for conf. interval', default=99, choices=[95, 99])
    parser.add_argument('--hw_file', type=str, help='which hardware file to use')
    parser.add_argument('--hardware', type=str, help='which hardware was used for benchmarks', default="9354P")
    parser.add_argument('--tol', type=float, help='error tolerance for declaring convergence', default=0.1)

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir1):
        sys.exit(f"Error: data_dir1 does not exist or is not a directory: {args.data_dir}")
    if not os.path.isdir(args.data_dir2):
        sys.exit(f"Error: data_dir2 does not exist or is not a directory: {args.data_dir}")
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
