import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import numpy as np
import sys
import os
import json

from roofline import Hardware

from plot import compute_PP, lighten, compute_PP_vs_n, compute_est_rel_err


####################################################################################
####################################################################################
    # PLOTTING SCRIPT FOR DEVELOPMENT ONLY
    # FINAL PLOTS WERE ALL OBTAINED WITH NOTEBOOKS IN "combined_plots"
    # SOME PLOTTING FUNCTIONS MAY BE OUTDATED
####################################################################################
####################################################################################


def get_krylov_bound(Tvals, nval):

    Hankel=np.zeros((nval,nval))
    for k in range(nval):
        for l in range(nval):
            Hankel[k,l]=Tvals[k+l+1]
        
    bvec=np.zeros(nval)
    for k in range(nval):
        bvec[k]=Tvals[k]

    # doing it like this is really important for numerical stability
    # inverting the Hankel matrix suffers from severe instabilities otherwise
    # (for non-tiny order)
    x = np.linalg.solve(Hankel, bvec)
    return np.real(bvec @ x)

def load_data(args, data_dir):
    # read in data_dir/*.npz
    data = {}
    qubits = []
    for npz_file in sorted(Path(data_dir).glob("*.npz")):
        tmp = dict(np.load(npz_file))
        for key in tmp:
            if type(tmp[key]) == np.ndarray and tmp[key].shape == (1,):
                tmp[key] = tmp[key][0]
        assert tmp["num_qubits"] not in data, "Only one file per num_qubits allowed"
        data[tmp["num_qubits"]] = tmp
        qubits.append(tmp["num_qubits"])

    # assert same parameters
    for idx, num_qubits in enumerate(data):
        if idx == 0:
            n_star = data[num_qubits]["n_star"]
            order = data[num_qubits]["order"]
            num_meas = data[num_qubits]["num_meas"]
            num_runs = data[num_qubits]["num_runs"]
            num_data_points = data[num_qubits]["num_data_points"]
        with open(f"{data_dir}/frequencies", "r") as f:
            lines = f.read().splitlines()
        if lines[idx + 1] != '':
            data[num_qubits]["freq"] = float(lines[idx + 1])            
        else:
            data[num_qubits]["freq"] = 0            
        assert n_star == data[num_qubits]["n_star"]
        assert order == data[num_qubits]["order"]
        assert num_meas == data[num_qubits]["num_meas"]
        assert num_runs == data[num_qubits]["num_runs"]
        assert num_data_points == data[num_qubits]["num_data_points"]

    min_qb = int(qubits[0])
    num_meas = int(data[min_qb]["num_meas"])
    num_runs = int(data[min_qb]["num_runs"])       
    num_data_points = int(data[min_qb]["num_data_points"])       

    return data, np.array(qubits, dtype=int), n_star, np.arange(0, order, dtype=int), num_meas, num_runs, num_data_points


def extract_data(args, in_data, qubits, n_star, orders, num_meas, num_runs, num_data_points, compute_DICI=True):
    assert type(num_meas) == int
    assert type(num_runs) == int
    assert type(num_data_points) == int
    assert 2 * n_star - 1 == orders[-1]
    assert 2 * n_star == len(orders)

    len_qubits = len(qubits)
    len_orders = len(orders)

    data = {
        "qubits": qubits,
        "orders": orders,
        "n_star": n_star,
        "num_meas": num_meas,
        "num_runs": num_runs,
        "num_data_points": num_data_points,
    }

    # shape = (len_qubits, 1)
    data["QFI"] = np.zeros(len_qubits, dtype=float)
    data["RE"] = np.zeros(len_qubits, dtype=float)
    data["runtimes"] = np.zeros(len_qubits, dtype=float)
    data["runtimesDSC"] = np.zeros(len_qubits, dtype=float)
    data["runtimesDSC_all"] = np.zeros([len_qubits, num_runs], dtype=float)
    data["num_threads"] = np.zeros(len_qubits, dtype=int)
    data["num_ranks"] = np.ones(len_qubits, dtype=int)
    data["vector_width"] = np.zeros(len_qubits, dtype=int)
    data["rmMB"] = np.zeros(len_qubits, dtype=float)
    data["rsptMB"] = np.zeros(len_qubits, dtype=float)
    data["rfMF"] = np.zeros(len_qubits, dtype=float)
    data["rmtMB"] = np.zeros(len_qubits, dtype=float)
    data["AP"] = np.zeros(len_qubits, dtype=float)
    data["PP"] = np.zeros(len_qubits, dtype=float)
    data["freq"] = np.zeros(len_qubits, dtype=float)

    # shape = (len_qubits, len_orders)
    data["Tvals"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["est_Tvals"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERR"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERR_max"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["RE"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["RE_max"] = np.zeros([len_qubits, len_orders], dtype=float)

    # shape = (len_qubits, len_orders, num_runs)
    data["DICI"] = np.zeros([len_qubits, len_orders, num_runs], dtype=float)

    # shape = (len_qubits, len_orders, num_data_points)
    data["inter_tidx"] = np.zeros([len_qubits, num_data_points], dtype=int)
    data["est_inter_Tvals"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_ERR_in_sigma"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_RE"] = np.zeros([len_qubits, num_data_points], dtype=float)
    data["inter_QFI"] = np.zeros([len_qubits, num_data_points], dtype=float)

    data["krylov_bounds"] = np.zeros([len_qubits, n_star], dtype=float)
    data["krylov_bounds_RE"] = np.zeros([len_qubits, n_star], dtype=float)
    
    for idx, n in enumerate(qubits):
        data["running"] = True
        if in_data[n]["inter_tidx"][-1] == 0:
            data["running"] = False

        QFI = np.tile(in_data[n]["QFI"], num_runs)
        inter_QFI = np.tile(in_data[n]["QFI"], num_data_points)
        inter_QFI = np.tile(inter_QFI, num_runs).reshape(num_runs, num_data_points)
        est_Tvals = in_data[n]["est_Tvals"].reshape(num_runs, len_orders)
        krylov_bounds = np.zeros([num_runs, n_star], dtype=float)
        krylov_bounds_RE = np.zeros([num_runs, n_star], dtype=float)
        for run in range(num_runs):
            for n_star_idx in range(n_star):
                krylov_bounds[run, n_star_idx] = get_krylov_bound(est_Tvals[run], n_star_idx + 1)
                krylov_bounds_RE[run, n_star_idx] = np.abs(krylov_bounds[run, n_star_idx] - in_data[n]["QFI"])/in_data[n]["QFI"]
        data["krylov_bounds"][idx] = np.median(krylov_bounds, axis=0)
        data["krylov_bounds_RE"][idx] = np.median(krylov_bounds_RE, axis=0)
        data["RE"][idx] = np.median(krylov_bounds_RE[:,-1], axis=0)

        if data["running"]:
            est_inter_Tvals = in_data[n]["est_inter_Tvals"].reshape(num_runs, num_data_points, len_orders)
            inter_tidx = in_data[n]["inter_tidx"]
            # inter_stderrs = in_data[n]["est_inter_std_errs"].reshape(num_runs, num_data_points, len_orders)
            # inter_z_sigma = args.CI_fac * inter_stderrs
            # est_inter_rel_err = compute_est_rel_err(est_inter_Tvals, inter_z_sigma)
            est_inter_QFI = np.zeros([num_runs, num_data_points], dtype=float)
            for run in range(num_runs):
                for dpt in range(num_data_points):
                    est_inter_QFI[run, dpt] = get_krylov_bound(est_inter_Tvals[run, dpt], n_star)

        DSC_only_runtime = np.median(in_data[n]["runtimesDSC"]) # nasty

        if data["running"]:
            inter_errs_signed = inter_QFI - est_inter_QFI
            inter_errs_signed[inter_errs_signed == 0] = np.nan
            inter_errs = np.abs(inter_errs_signed)
            inter_rel_errs = inter_errs/np.abs(inter_QFI)

        data["runtimes"][idx] = in_data[n]["runtime"]
        data["runtimesDSC"][idx] = DSC_only_runtime
        data["runtimesDSC_all"][idx] = in_data[n]["runtimesDSC"]
        data["num_threads"][idx] = in_data[n]["num_threads"]
        data["vector_width"][idx] = in_data[n]["vector_width"]
        data["rmMB"][idx] = in_data[n]["rmMB"] # required memory in MB
        data["rsptMB"][idx] = in_data[n]["rsptMB"] # required array data (wo cs data) in MB 
        data["rfMF"][idx] = in_data[n]["rfMF"] # performed flops
        data["rmtMB"][idx] = in_data[n]["rmtMB"] # idk
        data["freq"][idx] = in_data[n]["freq"] * 1000 # GHz

        data["QFI"][idx] = in_data[n]["QFI"]

        if data["running"]:
            data["inter_tidx"][idx] = np.copy(inter_tidx)
            data["inter_RE"][idx] = np.median(inter_rel_errs, axis=0)
            data["inter_QFI"][idx] = np.median(est_inter_QFI, axis=0)

    # add hw data
    with open(args.hw_file) as f:
        data["HW"] = json.load(f)[args.hardware]    

    if np.all(data["freq"] == 0):
        for idx, n in enumerate(qubits):      
            data["freq"][idx] = data["HW"]["freq"]  

    # used cores
    cores = data["num_threads"].copy()
    cores[cores > data["HW"]["cores"]] = data["HW"]["cores"]        
    data["cores"] = cores

    # peak performance in MFLOPs
    PP = compute_PP(cores=data["HW"]["cores"], 
                            freq=data["freq"], 
                            Vsize=data["HW"]["max_Vsize"], 
                            floatsize=args.floatsize, 
                            ILP_fac=data["HW"]["ILP_factor_FMA"],
                            FMA_fac=2,
                            nodes=max(data["num_ranks"]))       

    # compute achieved performance for each n
    if data["running"] == True:
        kmax = 2 * n_star + 1
        Ns = 2**np.array(data["qubits"])     
        AP = 14 * (n_star + 1) * Ns * data["num_meas"] * (kmax - 1 + data["qubits"]) * data["qubits"] * Ns \
            + 6 * (n_star + 1) * Ns * data["num_meas"] * (kmax - 1 + data["qubits"]) * Ns # total flops
        AP = AP / 1e6 # Mflops
        AP = AP / data["runtimesDSC"] # MFLOPs  

    else:
        kmax = 2 * n_star + 1
        Ns = 2**np.array(data["qubits"])     
        AP = 14 * Ns * data["num_meas"] * data["qubits"] * Ns \
            + 6 * Ns * data["num_meas"] * Ns # total flops
        AP = AP / 1e6 # Mflops
        AP = AP / data["runtimesDSC"] # MFLOPs
        
        # memory/ cache usage
        x = np.array(data["rsptMB"]).copy()
        SMT = data["num_threads"] > data["HW"]["cores"]
        x[SMT] = x[SMT] * 2

        data["pk_per_core"] = x / 2
        data["xy_per_core"] = x / 2
   
    data["AP"] = AP
    data["PP"] = PP          

    data["total_mem"] = data["rmMB"].copy()    

    return data

def log_run_details(args, data, data_dir, name="info.txt", print_state=True):
    with open(f"{args.out_dir}/{name}", "w") as f:
        print(f"path={data_dir}", file=f)
        if print_state:
            print(f"state={args.state}", file=f)
        print(f"qubits={data['qubits']}\norders={data['orders']}", file=f)
        print(f"HW={data['HW']['cpu_name']}", file=f)
        print(f"num_meas={data['num_meas']}, num_runs={data['num_runs']}, num_data_points={data['num_data_points']}", file=f)

def plot_RE_vs_T_fixed_m(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])] if data["num_data_points"] < 30 else None      
        ax.loglog(data["inter_tidx"][idx], data["inter_RE"][idx], c=c, marker=marker, label=f"n={n}")
    ax.set_xlabel("num_meas")
    ax.set_ylabel("RE")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    # min_RE_to_plot = 1e-4
    # max_RE_to_plot = 1e9
    # min_T = 1e3

    # min_RE = np.min(data["inter_RE"][:, midx])
    # max_RE = np.max(data["inter_RE"][:, midx])
    # # min_RE = min_RE_to_plot if min_RE < min_RE_to_plot else None
    # min_RE = min_RE / 2 if not np.isnan(min_RE) else min_RE_to_plot
    # max_RE = max_RE_to_plot if max_RE > max_RE_to_plot or np.isnan(max_RE) else max_RE * 2
    # ax.set_ylim(ymin=min_RE, ymax=max_RE)      
    # ax.set_xlim(xmin=min_T, xmax=None)     

def plot_RE_vs_n(ax, args, data, title=True):
    idx = 0
    c = data["colors"][idx%len(data["colors"])]
    marker = data["markers"][idx%len(data["markers"])]     
    ax.semilogy(data["qubits"], data["RE"], c=c, marker=marker)
    ax.set_xlabel("n")
    ax.set_ylabel("RE")
    # ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)


def plot_runtime_vs_n(ax, args, data, title=True):
    ax.semilogy(data["qubits"], data["runtimes"], marker="v", label="runtime")
    ax.set_xlabel("n")
    ax.set_ylabel("Runtime [s]")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def plot_comp_HW_usage_vs_n(ax, args, data, title=True):
    ax.semilogy(data["qubits"], data["vector_width"], marker="o", label="vector-width")
    ax.semilogy(data["qubits"], data["cores"], marker="s", label="#cores")
    ax.semilogy(data["qubits"], data["num_threads"], marker="v", label="#threads")
    print(f"num_ranks: {data['num_ranks']}")
    ax.set_xlabel("n")
    ax.set_ylabel("#cores/ #threads/ #nodes/ vector-width [B]")
    if title:
        ax.set_title(f"Used number of threads, number of cores,\nnodes, and vector-widths in bytes over n")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
    # for xx in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    #     ax.axhline(y=xx, linestyle="--", c="grey", alpha=0.6)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def plot_HW_util(ax, args, data, title=True):
    achievable_PP = compute_PP_vs_n(data, args)
    ax.plot(data["qubits"], 100 * data["AP"]/data["PP"], marker="v", label="HW util total")
    ax.plot(data["qubits"], 100 * data["AP"]/achievable_PP, marker="v", label="HW util per core")
    ax.plot(data["qubits"], 100 * data["freq"]/data["HW"]["max_freq"], marker="v", label="% of max. freq")
    ax.axhline(y=100, linestyle="--", c="black", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("%")
    if title:
        ax.set_title(f"Hardware-utilization over n with HW {data['HW']['cpu_name']}\nPP = (#cores) * (base_freq) * (data_path_width)\n/ (float_size) * (#fma_pipes) * 2 (fma)")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)    
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)  

# ---------------------- main ----------------------
def run_all_plots(args):

    # extract data
    in_data, qubits, n_star, orders, num_meas, num_runs, num_data_points = load_data(args, args.data_dir)
    data = extract_data(args, in_data, qubits, n_star, orders, num_meas, num_runs, num_data_points, compute_DICI=(args.version.startswith("running")))

    log_run_details(args, data, args.data_dir)

    # plot globals
    data["colors"] = [x for x in mcolors.TABLEAU_COLORS.keys()]
    data["markers"] = ["o", "v", "p", "*", "s", "X"]
    data["linestyles"] = ["-", ":", "-."]
    data["lw"] = 1

    # dirs
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(single_plt_dir, exist_ok=True)    

    # ----------------------- plots ----------------------- #

    if data["running"]:
        # --- plots over num_meas --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_RE_vs_T_fixed_m(ax, args, data, title=False)

        fg.savefig(f"{single_plt_dir}/RE_vs_T.png")
        fg.savefig(f"{single_plt_dir}/RE_vs_T.pdf", bbox_inches="tight")          
        plt.close(fg)
    else:
        # --- plots over num_meas --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_RE_vs_n(ax, args, data, title=False)

        fg.savefig(f"{single_plt_dir}/RE_vs_n.png")
        fg.savefig(f"{single_plt_dir}/RE_vs_n.pdf", bbox_inches="tight")          
        plt.close(fg)        

    # --- runtimes [s] over n --- #
    # single
    fg, ax = plt.subplots(layout="constrained")
    plot_runtime_vs_n(ax, args, data, title=False)
    
    fg.savefig(f"{single_plt_dir}/runtimes_vs_n.png")
    fg.savefig(f"{single_plt_dir}/runtimes_vs_n.pdf", bbox_inches="tight") 
    # --- runtimes [s] over n --- #

    # --- HW usage over n --- #
    # single
    fg, ax = plt.subplots(layout="constrained")
    plot_comp_HW_usage_vs_n(ax, args, data, title=False)
    
    fg.savefig(f"{single_plt_dir}/HW_usage_vs_n.png")
    fg.savefig(f"{single_plt_dir}/HW_usage_vs_n.pdf", bbox_inches="tight") 
    # --- HW usage over n --- #

    # --- HW util over n --- #
    # single
    fg, ax = plt.subplots(layout="constrained")
    plot_HW_util(ax, args, data, title=False)
    
    fg.savefig(f"{single_plt_dir}/HW_util_vs_n.png")
    fg.savefig(f"{single_plt_dir}/HW_util_vs_n.pdf", bbox_inches="tight") 
    # --- HW util over n --- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance report of moments runs with different qubits.')
    parser.add_argument('--data_dir', type=str, help='dir to parse for data')
    parser.add_argument('--out_dir', type=str, help='dir to store plots in')
    parser.add_argument('--state', type=str, help='name of state of data')
    parser.add_argument('--CI_perc', type=int, help='percentage to use for conf. interval', default=99, choices=[95, 99])
    parser.add_argument('--hw_file', type=str, help='which hardware file to use')
    parser.add_argument('--hardware', type=str, help='which hardware was used for benchmarks', default="9354P")
    parser.add_argument('--tol', type=float, help='error tolerance for declaring convergence', default=0.1)
    parser.add_argument('--julia', type=int, help='whether we are plotting julia results', default=0)
    parser.add_argument('--version', type=str, help='whether we are plotting running or batched', default="running")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"Error: data_dir does not exist or is not a directory: {args.data_dir}")

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

    args.strict_tol = args.tol / 10
    
    print(f"Plotting...")

    run_all_plots(args)
