import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import numpy as np
import sys
import os
import json

from plot import load_data, extract_data, log_run_details, lighten


####################################################################################
####################################################################################
    # PLOTTING SCRIPT FOR DEVELOPMENT ONLY
    # FINAL PLOTS WERE ALL OBTAINED WITH NOTEBOOKS IN "combined_plots"
    # SOME PLOTTING FUNCTIONS MAY BE OUTDATED
####################################################################################
####################################################################################


def cmp_RE_vs_order(ax, args, data, data_bs, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]        
        ax.semilogy(data["orders"], data["RE"][idx], c=c, marker=marker, label=f"n={n}, running")
        ax.semilogy(data["orders"], data_bs["RE"][idx], c=lighten(c), marker=marker, label=f"n={n}, batched")
    ax.set_xlabel("order")
    ax.set_ylabel("RE")
    if title:
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof moment estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_RE_to_plot = 1e2
    max_RE = max(np.max(data["RE"]), np.max(data_bs["RE"]))
    if max_RE > max_RE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_RE_to_plot)     

def cmp_RE_frac_vs_order(ax, args, data, data_bs, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]        
        ax.semilogy(data["orders"][-4:], data_bs["RE"][idx][-4:]/data["RE"][idx][-4:], c=c, marker=marker, label=f"n={n}")
    ax.axhline(y=(2)**(1/2), linestyle="--", c="slategrey", alpha=0.75, label=r"$\sqrt{2}$")
    ax.axhline(y=(np.e)**(1/2), linestyle="--", c="black", alpha=0.75, label=r"$ \sqrt{e}$")
    ax.set_xlabel("order")
    ax.set_ylabel("RE")
    if title:
        ax.set_title(f"Median improvement in rel. error (RE) of running\nto batched algorithm over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)       

def cmp_runtime_vs_n(ax, args, data, data_bs, title=True):
    ax.plot(data["qubits"], data_bs["runtimes"]/data["runtimes"] , marker="v", label="speedup")
    ax.axhline(y=1, linestyle="--", c="slategrey", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("SU")
    if title:
        ax.set_title(f"Speedup of running to batched of median runtime for estimating all moments\nup to order {data['orders'][-1]} over n\nwith {data_bs["num_batches"]} batches")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

# ---------------------- main ----------------------
def run_all_plots(args):

    # extract data
    in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points = load_data(args, args.data_dir1)
    in_data_bs, qubits2, orders2, PT2, var_offset2, num_meas2, num_runs2, num_data_points2 = load_data(args, args.data_dir2)

    assert np.all(qubits == qubits2)
    assert np.all(orders == orders2)
    assert PT == PT2
    assert var_offset == var_offset2
    assert num_meas == num_meas2
    assert num_runs == num_runs2
    assert num_data_points == num_data_points2

    data = extract_data(args, in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points)
    data_bs = extract_data(args, in_data_bs, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points, True)

    log_run_details(args, data, args.data_dir1, name="info_1.txt", print_state=False)
    log_run_details(args, data_bs, args.data_dir2, name="info_2.txt", print_state=False, julia=True)

    # plot globals
    data["colors"] = [x for x in mcolors.TABLEAU_COLORS.keys()]
    data["markers"] = ["o", "v", "p", "*", "s", "X"]
    data["linestyles"] = ["-", ":", "-."]
    data["lw"] = 1

    data_bs["colors"] = [lighten(x) for x in mcolors.TABLEAU_COLORS.keys()]
    data_bs["markers"] = ["o", "v", "p", "*", "s", "X"]
    data_bs["linestyles"] = ["-", ":", "-."]
    data_bs["lw"] = 1

    # dirs
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(single_plt_dir, exist_ok=True)    


    # ----------------------- plots ----------------------- #

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_RE_vs_order(ax, args, data, data_bs, title=True)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/RE_vs_order.png")
    fg.savefig(f"{single_plt_dir}/RE_vs_order.pdf", bbox_inches="tight")
    # --- rel. err over order --- #

    # --- rel. err frac over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_RE_frac_vs_order(ax, args, data, data_bs, title=True)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.png")
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.pdf", bbox_inches="tight")
    # --- rel. err frac over order --- #    

    # --- runtimes [s] over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_runtime_vs_n(ax, args, data, data_bs, title=True)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/runtimes_vs_n.png")
    fg.savefig(f"{single_plt_dir}/runtimes_vs_n.pdf", bbox_inches="tight")
    # --- runtimes [s] over n --- #

    # ----------------------- plots ----------------------- #


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
