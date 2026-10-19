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


####################################################################################
####################################################################################
    # PLOTTING SCRIPT FOR DEVELOPMENT ONLY
    # FINAL PLOTS WERE ALL OBTAINED WITH NOTEBOOKS IN "combined_plots"
    # SOME PLOTTING FUNCTIONS MAY BE OUTDATED
####################################################################################
####################################################################################


# returns Flops in unit that freq is given in
def compute_PP(cores, freq, Vsize, floatsize, ILP_fac, FMA_fac, nodes=1):
    return cores * freq * (Vsize / floatsize) * ILP_fac * FMA_fac * nodes

def compute_PP_vs_n(data, args):
    # possible FLOPs over n
    PP_vs_n = np.zeros(len(data["qubits"]), dtype=np.float64)
    for idx, n in enumerate(data["qubits"]):
        PP_vs_n[idx] = compute_PP(cores=data["cores"][idx], 
                            freq=data["freq"][idx], 
                            Vsize=data["HW"]["max_Vsize"], 
                            floatsize=args.floatsize, 
                            ILP_fac=data["HW"]["ILP_factor_FMA"],
                            FMA_fac=2,
                            nodes=data["num_ranks"][idx])
    return PP_vs_n

def lighten(color, amount=0.5):
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )

def compute_est_rel_err(est_moments, z_sigma):
    corr_est_rel_err = ((est_moments > 0) & (est_moments - z_sigma > 0))
    corr_est_rel_err = corr_est_rel_err | ((est_moments < 0) & (est_moments + z_sigma < 0))

    est_rel_err = z_sigma / np.abs( est_moments - np.sign(est_moments) * z_sigma )
    est_rel_err[~corr_est_rel_err] = np.inf

    return est_rel_err

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
            order = data[num_qubits]["order"]
            PT = data[num_qubits]["PT"]
            var_offset = data[num_qubits]["var_offset"]
            num_meas = data[num_qubits]["num_meas"]
            num_runs = data[num_qubits]["num_runs"]
            num_data_points = data[num_qubits]["num_data_points"]
        with open(f"{data_dir}/frequencies", "r") as f:
            lines = f.read().splitlines()
        if lines[idx + 1] != '':
            data[num_qubits]["freq"] = float(lines[idx + 1])            
        else:
            data[num_qubits]["freq"] = 0            
        assert order == data[num_qubits]["order"]
        assert var_offset == data[num_qubits]["var_offset"]
        assert PT == data[num_qubits]["PT"]
        assert num_meas == data[num_qubits]["num_meas"]
        assert num_runs == data[num_qubits]["num_runs"]
        assert num_data_points == data[num_qubits]["num_data_points"]

    min_qb = int(qubits[0])
    PT = bool(data[min_qb]["PT"])
    var_offset = int(data[min_qb]["var_offset"])
    num_meas = int(data[min_qb]["num_meas"])
    num_runs = int(data[min_qb]["num_runs"])       
    num_data_points = int(data[min_qb]["num_data_points"])       

    return data, np.array(qubits, dtype=int), np.arange(2, order + 1, dtype=int), PT, var_offset, num_meas, num_runs, num_data_points


def extract_data(args, in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points, julia=False, compute_DICI=True):
    assert type(PT) == bool
    assert type(var_offset) == int
    assert type(num_meas) == int
    assert type(num_runs) == int
    assert type(num_data_points) == int

    len_qubits = len(qubits)
    len_orders = len(orders)

    data = {
        "qubits": qubits,
        "orders": orders,
        "PT": PT,
        "var_offset": var_offset,
        "num_meas": num_meas,
        "num_runs": num_runs,
        "num_data_points": num_data_points,
    }

    if julia:
        data["num_batches"] = in_data[2]["num_batches"]

    data["running"] = True

    # shape = (len_qubits, 1)
    data["runtimes"] = np.zeros(len_qubits, dtype=float)
    data["runtimesDSC"] = np.zeros(len_qubits, dtype=float)
    data["runtimesDSC_all"] = np.zeros([len_qubits, num_runs], dtype=float)
    data["num_threads"] = np.zeros(len_qubits, dtype=int)
    data["num_ranks"] = np.zeros(len_qubits, dtype=int)
    data["vector_width"] = np.zeros(len_qubits, dtype=int)
    data["rmMB"] = np.zeros(len_qubits, dtype=float)
    data["rsptMB"] = np.zeros(len_qubits, dtype=float)
    data["rfMF"] = np.zeros(len_qubits, dtype=float)
    data["rmtMB"] = np.zeros(len_qubits, dtype=float)
    data["freq"] = np.zeros(len_qubits, dtype=float)

    # shape = (len_qubits, len_orders)
    data["moments"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["est_moments"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERR"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERR_max"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["RE"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["RE_max"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["CI"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["CIOE"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["CIOE_min"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERE"] = np.zeros([len_qubits, len_orders], dtype=float)
    data["ERE_max"] = np.zeros([len_qubits, len_orders], dtype=float)

    # shape = (len_qubits, len_orders, num_runs)
    data["DICI"] = np.zeros([len_qubits, len_orders, num_runs], dtype=float)

    # shape = (len_qubits, len_orders, num_data_points)
    data["inter_tidx"] = np.zeros([len_qubits, num_data_points], dtype=int)
    data["est_inter_moments"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_CI"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_CIOE"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_ERR_in_sigma"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_RE"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    data["inter_ERE"] = np.zeros([len_qubits, len_orders, num_data_points], dtype=float)
    
    for idx, n in enumerate(qubits):
        skip_o1 = 1 # we intentionally skip order 1 here
        moments = np.tile(in_data[n]["moments"][skip_o1:], num_runs).reshape(num_runs, len_orders)
        inter_moments = np.tile(in_data[n]["moments"][skip_o1:], num_data_points)
        inter_moments = np.tile(inter_moments, num_runs).reshape(num_runs, num_data_points, len_orders)
        est_moments = in_data[n]["est_moments"].reshape(num_runs, len_orders + 1)
        stderrs = in_data[n]["est_std_errs"].reshape(num_runs, len_orders + 1)[:, skip_o1:]

        if np.all(stderrs == 0):
            data["running"] = False

        if data["num_data_points"] > 0 and data["running"]:
            est_inter_moments = in_data[n]["est_inter_moments"].reshape(num_runs, num_data_points, len_orders + 1)[:, :, skip_o1:]
            inter_tidx = in_data[n]["inter_tidx"]
            inter_stderrs = in_data[n]["est_inter_std_errs"].reshape(num_runs, num_data_points, len_orders + 1)[:, :, skip_o1:]
            inter_z_sigma = args.CI_fac * inter_stderrs
            est_inter_rel_err = compute_est_rel_err(est_inter_moments, inter_z_sigma)

        distances = moments - est_moments[:, skip_o1:]
        errs = np.abs(moments - est_moments[:, skip_o1:])
        errs[errs == 0] = np.nan
        rel_errs = errs/np.abs(moments)
        z_sigma = args.CI_fac * stderrs
        est_rel_err = compute_est_rel_err(est_moments[:, skip_o1:], z_sigma)
        DSC_only_runtime = np.median(est_moments[:,0]) # nasty

        if data["num_data_points"] > 0 and data["running"]:
            inter_errs_signed = inter_moments - est_inter_moments
            inter_errs_signed[inter_errs_signed == 0] = np.nan
            inter_errs = np.abs(inter_errs_signed)
            inter_rel_errs = inter_errs/np.abs(inter_moments)

        data["runtimes"][idx] = in_data[n]["runtime"]
        data["runtimesDSC"][idx] = DSC_only_runtime
        data["runtimesDSC_all"][idx] = est_moments[:,0]
        data["num_threads"][idx] = in_data[n]["num_threads"]
        data["num_ranks"][idx] = in_data[n]["num_ranks"]
        data["vector_width"][idx] = in_data[n]["vector_width"]
        data["rmMB"][idx] = in_data[n]["rmMB"] # required memory in MB
        data["rsptMB"][idx] = in_data[n]["rsptMB"] # required array data (wo cs data) in MB 
        data["rfMF"][idx] = in_data[n]["rfMF"] # performed flops
        data["rmtMB"][idx] = in_data[n]["rmtMB"] # idk
        data["freq"][idx] = in_data[n]["freq"] * 1000 # GHz

        data["moments"][idx] = in_data[n]["moments"][skip_o1:]
        data["est_moments"][idx] = np.median(est_moments[:, skip_o1:], axis=0)
        data["ERR"][idx] = np.median(errs, axis=0) # ERRor
        data["ERR_max"][idx] = np.max(errs, axis=0)
        data["RE"][idx] = np.median(rel_errs, axis=0) # Relative Error
        data["RE_max"][idx] = np.max(rel_errs, axis=0)

        data["CI"][idx] = np.median(z_sigma, axis=0) # Confidence Interval
        data["CIOE"][idx] = np.median(z_sigma / errs, axis=0) # Confidence Interval OverEstimation
        data["CIOE_min"][idx] = np.min(z_sigma / errs, axis=0)

        data["ERE"][idx] = np.median(est_rel_err, axis=0) # Estimated Relative Error
        data["ERE_max"][idx] = np.max(est_rel_err, axis=0)

        if compute_DICI == True:
            data["DICI"][idx] = np.transpose(distances/(z_sigma)) # Distance In Confidence Interval (unit)

        if data["num_data_points"] > 0 and data["running"]:
            data["inter_tidx"][idx] = np.copy(inter_tidx)
            data["est_inter_moments"][idx] = np.transpose(np.median(est_inter_moments, axis=0))
            data["inter_CI"][idx] = np.transpose(np.median(inter_z_sigma, axis=0))
            data["inter_CIOE"][idx] = np.transpose(np.median(inter_z_sigma/inter_errs, axis=0))
            data["inter_ERR_in_sigma"][idx] = np.transpose(np.mean(inter_errs / inter_stderrs, axis=0))
            data["inter_RE"][idx] = np.transpose(np.median(inter_rel_errs, axis=0))
            data["inter_ERE"][idx] = np.transpose(np.median(est_inter_rel_err, axis=0))

    if PT:
        data["transposes"] = [0] * len_qubits
        for idx, n in enumerate(qubits):
            data["transposes"][idx] = in_data[n]["transposes"]


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
        kmax = data["orders"][-1]
        Ns = 2**np.array(data["qubits"])     
        AP = 14 * Ns * data["num_meas"] * kmax * data["qubits"] * Ns # total flops

        # flops for adding with row state, not considered here
        # AP += 6 * Ns * data["num_meas"] * kmax * Ns 

        AP = AP / 1e6 # Mflops
        AP = AP / data["runtimesDSC"] # MFLOPs  

        # memory/ cache usage
        x = np.array(data["rsptMB"]).copy()
        SMT = data["num_threads"] > data["HW"]["cores"]
        x[SMT] = x[SMT] * 2

        data["pk_per_core"] = x / (kmax + 2) * kmax
        data["xy_per_core"] = x / (kmax + 2) * 2  

    else:
        kmax = data["orders"][-1]
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

    data["total_mem"] = data["rmMB"].copy()
   
    data["AP"] = AP
    data["PP"] = PP           


    return data

def log_run_details(args, data, data_dir, name="info.txt", print_state=True, julia=False):
    with open(f"{args.out_dir}/{name}", "w") as f:
        print(f"path={data_dir}", file=f)
        if print_state:
            print(f"state={args.state}", file=f)
        print(f"qubits={data['qubits']}\norders={data['orders']}", file=f)
        print(f"HW={data['HW']['cpu_name']}", file=f)
        print(f"PT={data['PT']}, var_offset={data['var_offset']}, num_meas={data['num_meas']}, num_runs={data['num_runs']}, num_data_points={data['num_data_points']}", file=f)
        print(f"julia={julia}", file=f)
        if julia:
            print(f"num_batches={data["num_batches"]}", file=f)
        if data["PT"]:
            for idx, n in enumerate(data["qubits"]):
                print(f"n={n}: transposes={data['transposes'][idx]}", file=f)   


def plot_ERR_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]
        ax.semilogy(data["orders"], data["ERR"][idx], c=c, marker=marker, label=f"n={n}")
        if data["num_runs"] > 1:
            ax.semilogy(data["orders"], data["ERR_max"][idx], linestyle="--", linewidth=data["lw"], c=c)
    ax.set_xlabel("order")
    ax.set_ylabel("ERR")
    if title:
        ax.set_title(f"Median abs. error (ERR) to the correct moments\nof moment estimation over order for different n\nsolid lines ... median, dashed lines ... max")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)    

def plot_RE_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]               
        ax.semilogy(data["orders"], data["RE"][idx], c=c, marker=marker, label=f"n={n}")
        # if data["num_runs"] > 1:
        #     ax.semilogy(data["orders"], data["RE_max"][idx], linestyle="--", linewidth=data["lw"], c=c)
        # else:
        ax.semilogy(data["orders"], data["ERE"][idx], c=lighten(c))
    ax.set_xlabel("order")
    ax.set_ylabel("RE")
    if title:
        # lgd = "\nsolid lines ... median, dashed lines ... max" if data["num_runs"] > 1 else "dots ... point estimate, lines ... est. upper bound"
        lgd = "dots ... point estimate, lines ... est. upper bound, dashed lines ... max RE"
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof moment estimation over order for different n\n{lgd}")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_RE_to_plot = 1e2
    max_RE = np.max(data["RE"])
    if max_RE > max_RE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_RE_to_plot)       

def plot_ghz_RE_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        # even
        marker = data["markers"][idx%len(data["markers"])]      
        ax.semilogy(data["orders"][::2], data["RE"][idx][::2], c=c, marker=marker, label=f"n={n}, even")
        # if data["num_runs"] > 1:
        #     ax.semilogy(data["orders"][::2], data["RE_max"][idx][::2], linestyle="--", linewidth=data["lw"], c=c)     
        # else:
        ax.semilogy(data["orders"][::2], data["ERE"][idx][::2], c=lighten(c))

        # odd
        marker = data["markers"][(idx+1)%len(data["markers"])]      
        ax.semilogy(data["orders"][1::2], data["RE"][idx][1::2], c=c, marker=marker, label=f"n={n}, odd")
        # if data["num_runs"] > 1:
        #     ax.semilogy(data["orders"][1::2], data["RE_max"][idx][1::2], linestyle="--", linewidth=data["lw"], c=c)     
        # else:
        ax.semilogy(data["orders"][1::2], data["ERE"][idx][1::2], c=lighten(c))

    ax.set_xlabel("order")
    ax.set_ylabel("RE")
    if title:
        # lgd = "solid lines ... median, dashed lines ... max" if data["num_runs"] > 1 else "dots ... point estimate, lines ... est. upper bound"
        lgd = "dots ... point estimate, lines ... est. upper bound, dashed lines ... max RE"
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof moment estimation over order for different n\n{lgd}")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_RE_to_plot = 1e2
    max_RE = np.max(data["RE"])
    if max_RE > max_RE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_RE_to_plot)  

def plot_est_rel_err_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]        
        ax.semilogy(data["orders"], data["ERE"][idx], c=c, marker=marker, label=f"n={n}")
        # if data["num_runs"] > 1:
        #     ax.semilogy(data["orders"], data["ERE_max"][idx], linestyle="--", linewidth=data["lw"], c=c)
    ax.axhline(y=args.tol, linestyle="--", c="black", alpha=0.75, label=f"tol={args.tol}")
    ax.axhline(y=args.strict_tol, linestyle="--", c="grey", alpha=0.75, label=f"tol={args.strict_tol}")
    ax.set_xlabel("order")
    ax.set_ylabel("estimated rel. err.")
    if title:
        lgd = "solid lines ... median, dashed lines ... max" if data["num_runs"] > 1 else ""
        ax.set_title(f"Median estimated upper bound of RE obtained\nthrough variance estimation over order for different n\n{lgd}")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_ERE_to_plot = 1e2
    max_ERE = np.max(data["ERE"])
    if max_ERE > max_ERE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_ERE_to_plot)       


def plot_ghz_est_rel_err_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]

        # even
        marker = data["markers"][idx%len(data["markers"])]      
        ax.semilogy(data["orders"][::2], data["ERE"][idx][::2], c=c, marker=marker, label=f"n={n}, even")
        # odd
        marker = data["markers"][(idx+1)%len(data["markers"])]      
        ax.semilogy(data["orders"][1::2], data["ERE"][idx][1::2], c=c, marker=marker, label=f"n={n}, odd")

        # if data["num_runs"] > 1:
        #     ax.semilogy(data["orders"], data["ERE_max"][idx], linestyle="--", linewidth=data["lw"], c=c)
    ax.axhline(y=args.tol, linestyle="--", c="black", alpha=0.75, label=f"tol={args.tol}")
    ax.axhline(y=args.strict_tol, linestyle="--", c="grey", alpha=0.75, label=f"tol={args.strict_tol}")
    ax.set_xlabel("order")
    ax.set_ylabel("estimated rel. err.")
    if title:
        ax.set_title(f"Median estimated upper bound of RE obtained\nthrough variance estimation over order for different n")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_ERE_to_plot = 1e2
    max_ERE = np.max(data["ERE"])
    if max_ERE > max_ERE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_ERE_to_plot)   

def plot_CIOE_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]                 
        ax.semilogy(data["orders"], data["CIOE"][idx], c=c, marker=marker, label=f"n={n}")
        if data["num_runs"] > 1:
            ax.semilogy(data["orders"], data["CIOE_min"][idx], linestyle="--", linewidth=data["lw"], c=c)
    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75, label="1")
    ax.axhline(y=args.CI_fac, linestyle="--", c="black", alpha=0.75, label=f"{args.CI_fac}")
    ax.set_xlabel("order")
    ax.set_ylabel("Overestimation (CI/ERR)")
    if title:
        lgd = "solid lines ... median, dashed lines ... max" if data["num_runs"] > 1 else ""
        ax.set_title(f"Median overestimation of {args.CI_perc}% conf. interval (CI) to the true\nerror (ERR) of moment estimation over order for different n\n{lgd}")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

def plot_ghz_CIOE_vs_order(ax, args, data, title=True):
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        # even
        marker = data["markers"][idx%len(data["markers"])]  
        ax.semilogy(data["orders"][::2], data["CIOE"][idx][::2], c=c, marker=marker, label=f"n={n}, even")
        # odd
        marker = data["markers"][(idx+1)%len(data["markers"])]  
        ax.semilogy(data["orders"][1::2], data["CIOE"][idx][1::2], c=c, marker=marker, label=f"n={n}, odd")
        if data["num_runs"] > 1:
            ax.semilogy(data["orders"], data["CIOE_min"][idx], linestyle="--", linewidth=data["lw"], c=c)
    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75)
    ax.axhline(y=args.CI_fac, linestyle="--", c="black", alpha=0.75, label=f"{args.CI_fac}")
    ax.set_xlabel("order")
    ax.set_ylabel("Overestimation (CI/ERR)")
    if title:
        lgd = "solid lines ... median, dashed lines ... max" if data["num_runs"] > 1 else ""
        ax.set_title(f"Median overestimation of {args.CI_perc}% conf. interval (CI) to the true\nerror (ERR) of moment estimation over order for different n\n{lgd}")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)     

def plot_DICI_vs_order(ax, args, data, title=True):
    plt.hist(data["DICI"].flatten(), bins="auto", density=True)

    ax.axvline(x=1, linestyle="--", c="grey", alpha=0.75)
    ax.axvline(x=0, linestyle="--", c="black", alpha=0.75)
    ax.axvline(x=-1, linestyle="--", c="grey", alpha=0.75)
    ax.set_xlabel(r"deviation")
    ax.set_ylabel("density")
    if title:
        ax.set_title(f"Distance of true moment to est. moment in units of\n{args.CI_perc}% conf. interval (CI) over order for different n")
    # ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)     
    ax.set_xlim(xmin=-2, xmax=2)
    ax.set_xticks(
        [-1, 0, 1],
        [r"$-1.96\,\sigma$", "0", r"$1.96\,\sigma$"]
    )    

def plot_RE_vs_T_fixed_m(ax, args, data, title=True, order=2):
    midx = order - 2
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])] if data["num_data_points"] < 30 else None      
        ax.loglog(data["inter_tidx"][idx] + midx, data["inter_RE"][idx, midx], c=c, marker=marker, label=f"n={n}")
        ax.loglog(data["inter_tidx"][idx] + midx, data["inter_ERE"][idx, midx], c=lighten(c))
    ax.set_xlabel("num_meas")
    ax.set_ylabel("RE")
    if title:
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof order {order} moment estimation over #shots for different n\n median over {data["num_runs"]} runs")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    min_RE_to_plot = 1e-4
    max_RE_to_plot = 1e9
    min_T = 1e3

    min_RE = np.min(data["inter_RE"][:, midx])
    max_RE = np.max(data["inter_RE"][:, midx])
    # min_RE = min_RE_to_plot if min_RE < min_RE_to_plot else None
    min_RE = min_RE / 2 if not np.isnan(min_RE) else min_RE_to_plot
    max_RE = max_RE_to_plot if max_RE > max_RE_to_plot or np.isnan(max_RE) else max_RE * 2
    ax.set_ylim(ymin=min_RE, ymax=max_RE)      
    ax.set_xlim(xmin=min_T, xmax=None)     

def plot_RE_vs_T_fixed_n(ax, args, data, title=True, n=2):
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
        ax.loglog(data["inter_tidx"][nidx] + midx, data["inter_RE"][nidx, midx], lw=1, c=lighten(c), marker=marker)    
    iidx = 0
    for idx, order in enumerate(data["orders"]):
        # if (order & (order - 1)) != 0: # pow of 2
        #     continue
        if order not in plt_orders:
            continue
        iidx += 1
        midx = order - 2
        c = data["colors"][iidx%len(data["colors"])]
        marker = data["markers"][iidx%len(data["markers"])]# if data["num_data_points"] < 30 else None      
        ax.loglog(data["inter_tidx"][nidx] + midx, data["inter_ERE"][nidx, midx], c=c, marker=marker, label=f"k={order}")
    ax.set_xlabel("num_meas")
    ax.set_ylabel("RE")
    if title:
        ax.set_title(f"Median rel. error (RE) to the correct moments\nof n={n} moment estimation over #shots for different k\n median over {data["num_runs"]} runs")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    manual = False
    if manual:
        min_RE_to_plot = 1e-4
        max_RE_to_plot = 5e1
        min_T = 6e5

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

def plot_avg_ERR_in_sigma_vs_T(ax, args, data, title=True, n=None):
    if n != None:
        nidx = np.where(data["qubits"] == n)[0][0]
        ys = np.mean(data["inter_ERR_in_sigma"][nidx], axis=0)
    else:
        nidx = 0
        ys = np.mean(data["inter_ERR_in_sigma"], axis=(0,1))
    c = data["colors"][nidx%len(data["colors"])]
    marker = data["markers"][nidx%len(data["markers"])] if data["num_data_points"] < 30 else None        
    ax.semilogx(data["inter_tidx"][nidx], ys, c=c, marker=marker)
    ax.axhline(y=0, linestyle="--", c="black", alpha=0.75)
    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75)        
    ax.set_xlabel("num_meas")
    ax.set_ylabel(r"Avg. error [$\sigma$]")
    if title:
        ax.set_title(f"Median conf. interval over estimation (CIOE) of correct\n error of moment estimation over #shots\n median over {data["num_runs"]} runs")
    # ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    min_T = 1e3
    ax.set_xlim(xmin=min_T, xmax=None)     

def plot_inter_moments_vs_T_fixed_n(ax, args, data, title=True, n=2):
    nidx = np.where(data["qubits"] == n)[0][0]
    iidx = 0
    for idx, order in enumerate(data["orders"]):
        if order not in [32]:
            continue
        iidx += 1
        midx = order - 2
        c = data["colors"][iidx%len(data["colors"])]
        marker = data["markers"][iidx%len(data["markers"])] if data["num_data_points"] <= 10 else None               
        ax.loglog((data["inter_tidx"][nidx] + midx), data["est_inter_moments"][nidx, midx], c=c, marker=marker, label=f"k={order}")
        ax.axhline(y=data["moments"][nidx, midx], linestyle="--", c=c, alpha=0.75)
    ax.set_xlabel("num_meas")
    ax.set_ylabel("moments")
    if title:
        ax.set_title(f"Estimated moments over T of n={n} for different k\ndashed lines corr. to true moments\n median over {data["num_runs"]} runs")
    ax.legend(ncol=2, fontsize=5, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    min_T = 1e7
    ax.set_xlim(xmin=min_T, xmax=None)    
    ax.set_ylim(ymin=1e-41, ymax=1e-40)      


def plot_corr_est_moments_vs_order(ax, args, data, positives=True, negatives=False, title=True):
    assert (positives or negatives) and not (positives and negatives)
    num_tols = 3
    num_converged = [np.zeros(len(data["qubits"]), dtype=int) for _ in range(num_tols)]
    tols = [args.tol * 10**(-x) for x in range(0, num_tols)]
    title_str = "positive" if positives else "abs. value of negative"
    for idx, n in enumerate(data["qubits"]):
        c = data["colors"][idx%len(data["colors"])]
        marker = data["markers"][idx%len(data["markers"])]          
        moments = data["moments"][idx]
        pos_neg_mask = (moments >= 0) if positives else (moments <= 0)
        converged = [(data["ERE"][idx] <= tol) for tol in tols]
        mask = pos_neg_mask & converged[0]

        ax.plot(data["orders"][mask], np.abs(moments[mask]), c=c, lw=1, label=f"n={n}")
        point_est = np.abs(data["est_moments"][idx][mask])
        for i, tol in enumerate(tols):
            num_converged[i][idx] = np.sum(converged[i])
        yerr = data["CI"][idx][mask]
        ax.errorbar(data["orders"][mask], 
                    point_est, 
                    yerr=yerr,
                    fmt=marker,
                    capsize=5,
                    capthick=1.2,
                    linestyle="None",
                    markersize=4,
                    color=c,
                    ecolor=c
                    )
    ax.set_yscale('log')

    ax_inset = inset_axes(
        ax,
        width="30%",     
        height="30%",     
        loc="lower left",
        bbox_to_anchor=(0.08, 0.08, 1, 1),  
        bbox_transform=ax.transAxes
    )
    for i, tol in enumerate(tols):
        ax_inset.plot(data["qubits"], num_converged[i], marker=data["markers"][i], markersize=3, linewidth=0.9, label=f"tol={tol}")
    ax_inset.set_xticks(data["qubits"])
    if num_tols == 1:
        ax_inset.set_yticks(list(num_converged[0]))
        ax_inset.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_inset.grid(which="major", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax_inset.grid(which="minor", visible=False)
    ax_inset.set_xlabel("n", fontsize=7, labelpad=2)
    ax_inset.set_ylabel("#converged", fontsize=7, labelpad=2)
    ax_inset.tick_params(labelsize=6, pad=2)
    ax_inset.legend(ncol=2, fontsize=4, markerscale=0.6, handlelength=1.0)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.6)

    ax.set_xlabel("order")
    ax.set_ylabel("moment")
    if title:
        ax.set_title(f"Converged {title_str} estimated (dots; median; tol={args.tol})\nwith error bars and true moments (lines) over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)


def plot_runtime_vs_n(ax, args, data, title=True):
    trans_qb = 5
    plot_ref_line = True
    while trans_qb not in data["qubits"]:
        trans_qb += 1
        if trans_qb > np.max(data["qubits"]):
            plot_ref_line = False
            break
    if plot_ref_line:
        qb_idx = np.where(data["qubits"] == trans_qb)[0][0]
        two_n_scaling = data["qubits"]*2**data["qubits"] / (data["qubits"][0]*2**data["qubits"][0]) * data["runtimes"][0]
        four_n_scaling = data["qubits"]*4**data["qubits"] / (data["qubits"][qb_idx]*4**data["qubits"][qb_idx]) * data["runtimes"][qb_idx]
        ax.semilogy(data["qubits"], two_n_scaling, linestyle="--", c="grey", label=r"$n2^n$")
        ax.semilogy(data["qubits"][qb_idx - 1:], four_n_scaling[qb_idx - 1:], linestyle="--", c="black", label=r"$n4^n$")
    ax.semilogy(data["qubits"], data["runtimes"], marker="v", label="runtime")
    ax.set_xlabel("n")
    ax.set_ylabel("Runtime [s]")
    if title:
        ax.set_title(f"Median. runtime in seconds for estimating all moments\nup to order {data['orders'][-1]} over n")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def plot_comp_HW_usage_vs_n(ax, args, data, title=True):
    ax.semilogy(data["qubits"], data["vector_width"], marker="o", label="vector-width")
    ax.semilogy(data["qubits"], data["cores"], marker="s", label="#cores")
    ax.semilogy(data["qubits"], data["num_threads"], marker="v", label="#threads")
    ax.semilogy(data["qubits"], data["num_ranks"], marker="x", label="#nodes")
    # print(f"num_ranks: {data['num_ranks']}")
    ax.set_xlabel("n")
    ax.set_ylabel("#cores/ #threads/ #nodes/ vector-width [B]")
    if title:
        ax.set_title(f"Used number of threads, number of cores,\nnodes, and vector-widths in bytes over n")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    # for xx in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    #     ax.axhline(y=xx, linestyle="--", c="grey", alpha=0.6)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def plot_mem_HW_usage_vs_n(ax, args, data, title=True):

    ax.semilogy(data["qubits"], data["total_mem"], marker="v", label="Memory")
    ax.semilogy(data["qubits"], data["pk_per_core"], marker="v", label="p_k per core")
    ax.semilogy(data["qubits"],  data["xy_per_core"], marker="o", label="x, y per core")

    ax.axhline(y=data["HW"]["L1d"], linestyle="--", c="slategrey", alpha=0.75, label=f"L1d per core")
    ax.axhline(y=data["HW"]["L2"], linestyle="--", c="dimgrey", alpha=0.75, label=f"L2 per core")

    ax.set_xlabel("n")
    ax.set_ylabel("MB")
    if title:
        ax.set_title(f"Total required memory (meas. data + matrices)\nwith HW ({data['HW']['cpu_name']}) metrics")
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

def plot_building_batch_frac(ax, args, data, title=True):
    Ns = 2**np.array(data["qubits"])     
    bb = 14 * Ns * data["num_meas"] * data["qubits"] * Ns \
        + 6 * Ns * data["num_meas"] * Ns # total flops
    bb = bb / 1e6 # Mflops
    mb = 8 * 8**np.array(data["qubits"]) / 1e6
    ax.plot(data["qubits"], 100 * bb/(bb + mb), marker="v")
    ax.axhline(y=100, linestyle="--", c="black", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("Fraction of building batches [%]")
    if title:
        ax.set_title(f"Percentage of total flops made up of building batches\n num_meas = {data["num_meas"]}, m = {data["orders"][-1]}")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)    

def compute_AI(data):
    
    # transition n (regime 1 -> 2)
    trans_idx_1_2 = np.where(data["pk_per_core"] > data["HW"]["L2"])[0]
    if len(trans_idx_1_2) > 0:
        trans_idx_1_2 = trans_idx_1_2[0]
        n_trans_1_2 = data["qubits"][trans_idx_1_2]
    else:
        n_trans_1_2 = np.inf   

    # transition n (regime 2 -> 3)
    trans_idx_2_3 = np.where(data["xy_per_core"] > data["HW"]["L2"])[0]
    if len(trans_idx_2_3) > 0:
        trans_idx_2_3 = trans_idx_2_3[0]
        n_trans_2_3 = data["qubits"][trans_idx_2_3]
    else:
        n_trans_2_3 = np.inf    

    # reusable things
    kmax = data["orders"][-1]
    Ns = 2**np.array(data["qubits"]) 
    
    # compute arithmetic intensities
    # flops and mem traffics per measurement
    flops_SRM = kmax * data["qubits"] * Ns**2 * 14
    flops_add_norm = kmax * Ns**2 * 6
    mem_traf_meas_data = data["qubits"] * Ns * 8
    mem_traf_pk = kmax * Ns**2 * 2 * 16
    mem_traf_xy = kmax * data["qubits"] * Ns**2 * 2 * 16
    AI_1 = (flops_SRM + flops_add_norm)/(mem_traf_meas_data)
    AI_2 = (flops_SRM + flops_add_norm)/(mem_traf_meas_data + mem_traf_pk)
    AI_3 = (flops_SRM + flops_add_norm)/(mem_traf_meas_data + mem_traf_pk + mem_traf_xy)
    AI = np.zeros(len(data["qubits"]), dtype=np.float64)
    for idx, n in enumerate(data["qubits"]):
        if n > n_trans_2_3:
            AI[idx] = AI_2[idx]
        elif n > n_trans_1_2:
            AI[idx] = AI_1[idx]
        else:
            AI[idx] = AI_1[idx]         
    return AI

def compute_BWs(data):
    BWs = np.zeros(len(data["qubits"]), dtype=np.float64)
    for idx, n in enumerate(data["qubits"]):
        num_cores = data["cores"][idx]
        BWs[idx] = data["HW"]["stream_BW_multicore"][f"{num_cores}"]
    return BWs

def plot_rooflines(ax, data, PPs, BWs, fac, plot_label=True, plot_qb_label=True):
    arch_str = {8: "scalar", 16: "SSE", 32: "AVX2", 64: "AVX512"}

    done = False
    for idx, n in enumerate(data["qubits"]):
        num_cores = data["cores"][idx]
        c = data["colors"][idx%len(data["colors"])]
        largest_num_cores = (num_cores == np.max(data["cores"]))
        n_str = fr"$\geq${n}" if largest_num_cores else f"{n}"        

        if np.all(data["cores"] == 1) and n != data["qubits"][-1]:
            continue

        # L3 roof
        if num_cores == data["HW"]["cores"]:
            done = True
            num_CCX = data["HW"]["cores"] // data["HW"]["cores_per_ccx"]
            L3BW = data["HW"]["L3_BW"] * num_CCX
            HW = Hardware(PP=PPs[idx]/fac, BW=L3BW/fac)
            if plot_label:
                HW.plotRoof(ax, data["colors"][-1], label=f"L3 {num_CCX}x")
            else:
                HW.plotRoof(ax, data["colors"][-1])
            HW.plotCeiling(ax, data["colors"][-1])

        HW = Hardware(PP=PPs[idx]/fac, BW=BWs[idx]/fac)
        if plot_label and plot_qb_label:
            HW.plotRoof(ax, c, label=f"{n_str}qb, DDR5 BW, {arch_str[data['vector_width'][idx]]}, {num_cores} cores")
        elif plot_label and not plot_qb_label:
            HW.plotRoof(ax, c, label=f"DDR5 BW, {arch_str[data['vector_width'][idx]]}, {num_cores} cores")
        else:
            HW.plotRoof(ax, c)
        HW.plotCeiling(ax, c)

        if done:
            return

def scatter_roofline_data_points(ax, data, AI, fac):
    # scatter data points
    cidx = 0
    for idx, n in enumerate(data["qubits"]):
        num_cores = data["cores"][idx]

        # handle coloring
        SMT = (data["num_threads"][idx] > num_cores)
        if SMT:
            if data["num_threads"][idx - 1] >= num_cores:
                cidx -= 1
        c = data["colors"][cidx%len(data["colors"])]

        ax.scatter(AI[idx], data["AP"][idx]/fac, marker="v", color=c)
        cidx += 1

def plot_roofline_model(ax, args, data, title=True):
    fac = 1e3 # GFLOPs (M to G)

    AI = compute_AI(data)
    PPs = compute_PP_vs_n(data, args)
    BWs = compute_BWs(data)

    plot_rooflines(ax, data, PPs, BWs, fac)
    scatter_roofline_data_points(ax, data, AI, fac)

    ax.set_xlabel("Arithmetic Intensity")
    ax.set_ylabel("GFLOPs")
    if title:
        ax.set_title(f"Roofline model with computed peak perf. and measured\nDRAM bandwidths of {data['HW']['cpu_name']} processor")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=4, markerscale=0.6, handlelength=1.0)    

# ---------------------- main ----------------------
def run_all_plots(args):

    # extract data
    in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points = load_data(args, args.data_dir)
    data = extract_data(args, in_data, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points, julia=args.julia, compute_DICI=(args.version.startswith("running")))

    log_run_details(args, data, args.data_dir, julia=args.julia)

    # plot globals
    data["colors"] = [x for x in mcolors.TABLEAU_COLORS.keys()]
    data["markers"] = ["o", "v", "p", "*", "s", "X"]
    data["linestyles"] = ["-", ":", "-."]
    data["lw"] = 1

    # dirs
    multi_plt_dir = f"{args.out_dir}/multi"
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(multi_plt_dir, exist_ok=True)    
    os.makedirs(single_plt_dir, exist_ok=True)    

    # subplots
    # 3x3
    rows = 3
    cols = 3
    cell_w = 6 
    cell_h = 4.5
    dpi = 140
    figsize = (cols * cell_w, rows * cell_h)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, layout="constrained")
    axes = axes.flatten()
    max_plots = rows * cols
    ax_ctr = 0 

    # row 0
    rows0 = 1
    cols0 = 3
    cell_w0 = 6 
    cell_h0 = 4.5
    dpi0 = 140
    figsize0 = (cols0 * cell_w0, rows0 * cell_h0)

    fig0, axes0 = plt.subplots(rows0, cols0, figsize=figsize0, dpi=dpi0, layout="constrained")
    axes0 = axes0.flatten()
    max_plots0 = rows0 * cols0
    ax_ctr0 = 0 


    # ----------------------- plots ----------------------- #

    if not args.julia:
        if args.state == "ghz":
            # --- rel. err over order --- #
            # single
            fg, ax = plt.subplots(layout="constrained")
            plot_ghz_RE_vs_order(ax, args, data, title=False)
            
            fg.savefig(f"{single_plt_dir}/RE_vs_order.png")
            fg.savefig(f"{single_plt_dir}/RE_vs_order.pdf", bbox_inches="tight")

            # multi
            plot_ghz_RE_vs_order(axes[ax_ctr], args, data)
            ax_ctr += 1    
            plot_ghz_RE_vs_order(axes0[ax_ctr0], args, data)
            ax_ctr0 += 1            
            # --- rel. err over order --- #
        else:
            # --- rel. err over order --- #
            # single
            fg, ax = plt.subplots(layout="constrained")
            plot_RE_vs_order(ax, args, data, title=False)
            
            fg.savefig(f"{single_plt_dir}/RE_vs_order.png")
            fg.savefig(f"{single_plt_dir}/RE_vs_order.pdf", bbox_inches="tight")

            # multi
            plot_RE_vs_order(axes[ax_ctr], args, data)
            ax_ctr += 1    
            plot_RE_vs_order(axes0[ax_ctr0], args, data)
            ax_ctr0 += 1            
            # --- rel. err over order --- #

        if args.version.startswith("running"):
            if args.state == "ghz":
                # --- rel. uncertainty over n --- #
                # single
                fg, ax = plt.subplots(layout="constrained")
                plot_ghz_est_rel_err_vs_order(ax, args, data, title=False)
                
                fg.savefig(f"{single_plt_dir}/est_rel_err_vs_order.png")
                fg.savefig(f"{single_plt_dir}/est_rel_err_vs_order.pdf", bbox_inches="tight")

                # multi
                plot_ghz_est_rel_err_vs_order(axes[ax_ctr], args, data)
                ax_ctr += 1    
                plot_ghz_est_rel_err_vs_order(axes0[ax_ctr0], args, data)
                ax_ctr0 += 1          
                # --- rel. uncertainty over n --- #
            else:
                # --- rel. uncertainty over n --- #
                # single
                fg, ax = plt.subplots(layout="constrained")
                plot_est_rel_err_vs_order(ax, args, data, title=False)
                
                fg.savefig(f"{single_plt_dir}/est_rel_err_vs_order.png")
                fg.savefig(f"{single_plt_dir}/est_rel_err_vs_order.pdf", bbox_inches="tight")

                # multi
                plot_est_rel_err_vs_order(axes[ax_ctr], args, data)
                ax_ctr += 1    
                plot_est_rel_err_vs_order(axes0[ax_ctr0], args, data)
                ax_ctr0 += 1    
                # --- rel. uncertainty over n --- #


            if args.state == "ghz":
                # --- CI overestimation over order --- #
                # single
                fg, ax = plt.subplots(layout="constrained")
                plot_ghz_CIOE_vs_order(ax, args, data, title=False)
                
                fg.savefig(f"{single_plt_dir}/CIOE_vs_order.png")
                fg.savefig(f"{single_plt_dir}/CIOE_vs_order.pdf", bbox_inches="tight")

                # multi
                plot_ghz_CIOE_vs_order(axes[ax_ctr], args, data)
                ax_ctr += 1    
                plot_ghz_CIOE_vs_order(axes0[ax_ctr0], args, data)
                ax_ctr0 += 1            
                # --- CI overestimation over order --- #
            else:
                # --- CI overestimation over order --- #
                # single
                fg, ax = plt.subplots(layout="constrained")
                plot_CIOE_vs_order(ax, args, data, title=False)
                
                fg.savefig(f"{single_plt_dir}/CIOE_vs_order.png")
                fg.savefig(f"{single_plt_dir}/CIOE_vs_order.pdf", bbox_inches="tight")

                # multi
                plot_CIOE_vs_order(axes[ax_ctr], args, data)
                ax_ctr += 1    
                plot_CIOE_vs_order(axes0[ax_ctr0], args, data)
                ax_ctr0 += 1            
                # --- CI overestimation over order --- #

            # --- distance to CI over order --- #
            # single
            fg, ax = plt.subplots(layout="constrained")
            plot_DICI_vs_order(ax, args, data, title=False)
            
            fg.savefig(f"{single_plt_dir}/DICI_vs_order.png")
            fg.savefig(f"{single_plt_dir}/DICI_vs_order.pdf", bbox_inches="tight")          
            # --- distance to CI over order --- #

            if data["num_data_points"] > 0:
                # --- plots over num_meas --- #
                # single
                for order in data["orders"]:
                    fg, ax = plt.subplots(layout="constrained")
                    plot_RE_vs_T_fixed_m(ax, args, data, title=False, order=order)

                    trg_dir = f"{single_plt_dir}/RE_vs_T_fixed_m"
                    os.makedirs(trg_dir, exist_ok=True)    
                    fg.savefig(f"{trg_dir}/order_{order}.png")
                    fg.savefig(f"{trg_dir}/order_{order}.pdf", bbox_inches="tight")          
                    plt.close(fg)

                for n in data["qubits"]:
                    fg, ax = plt.subplots(layout="constrained")
                    plot_RE_vs_T_fixed_n(ax, args, data, title=False, n=n)

                    trg_dir = f"{single_plt_dir}/RE_vs_T_fixed_n"
                    os.makedirs(trg_dir, exist_ok=True)    
                    fg.savefig(f"{trg_dir}/n_{n}.png")
                    fg.savefig(f"{trg_dir}/n_{n}.pdf", bbox_inches="tight")          
                    plt.close(fg)

                fg, ax = plt.subplots(layout="constrained")
                plot_avg_ERR_in_sigma_vs_T(ax, args, data, title=False)                  
                fg.savefig(f"{single_plt_dir}/avg_ERR_in_sigma_vs_T.png")
                fg.savefig(f"{single_plt_dir}/avg_ERR_in_sigma_vs_T.pdf", bbox_inches="tight")          
                plt.close(fg)           
                for n in data["qubits"]:
                    fg, ax = plt.subplots(layout="constrained")
                    plot_avg_ERR_in_sigma_vs_T(ax, args, data, title=False, n=n)
                    trg_dir = f"{single_plt_dir}/avg_ERR_in_sigma_vs_T"
                    os.makedirs(trg_dir, exist_ok=True)                       
                    fg.savefig(f"{trg_dir}/n_{n}.png")
                    fg.savefig(f"{trg_dir}/n_{n}.pdf", bbox_inches="tight")          
                    plt.close(fg)               


                # for n in data["qubits"]:
                #     fg, ax = plt.subplots(layout="constrained")
                #     plot_inter_moments_vs_T_fixed_n(ax, args, data, title=True, n=n)

                #     trg_dir = f"{single_plt_dir}/moments_vs_T_fixed_n"
                #     os.makedirs(trg_dir, exist_ok=True)    
                #     fg.savefig(f"{trg_dir}/n_{n}.png")
                #     fg.savefig(f"{trg_dir}/n_{n}.pdf", bbox_inches="tight")          
                #     plt.close(fg)            
                # --- plots over num_meas --- #

            # finish multi-plot row 0
            tmp = "PT" if data["PT"] else ""
            fig0.suptitle(f"{tmp} moment estimation of state '{args.state}', num_meas={data['num_meas']:.3e}, var_offset={data['var_offset']}, num_samples={data['num_runs']}, m={data['orders'][-1]}")
            fig0.savefig(f"{multi_plt_dir}/{args.state}0.png")
            fig0.savefig(f"{multi_plt_dir}/{args.state}0.pdf", bbox_inches="tight")

        # --- corr. moments and median est. moments over order --- #
        # single
        # positives
        fg, ax = plt.subplots(layout="constrained")
        plot_corr_est_moments_vs_order(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/pos_true_est_moments_vs_order.png")
        fg.savefig(f"{single_plt_dir}/pos_true_est_moments_vs_order.pdf", bbox_inches="tight")

        # negatives
        fg, ax = plt.subplots(layout="constrained")
        plot_corr_est_moments_vs_order(ax, args, data, positives=False, negatives=True, title=False)
        
        fg.savefig(f"{single_plt_dir}/neg_true_est_moments_vs_order.png")    
        fg.savefig(f"{single_plt_dir}/neg_true_est_moments_vs_order.pdf", bbox_inches="tight")    

        # multi
        plot_corr_est_moments_vs_order(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- corr. moments and median est. moments over order --- #

        # --- runtimes [s] over n --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_runtime_vs_n(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/runtimes_vs_n.png")
        fg.savefig(f"{single_plt_dir}/runtimes_vs_n.pdf", bbox_inches="tight")

        # multi
        plot_runtime_vs_n(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- runtimes [s] over n --- #


        # --- comp. HW usage over n --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_comp_HW_usage_vs_n(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/comp_HW_usage_vs_n.png")
        fg.savefig(f"{single_plt_dir}/comp_HW_usage_vs_n.pdf", bbox_inches="tight")

        # multi
        plot_comp_HW_usage_vs_n(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- comp. HW usage over n --- #


        # --- mem. HW usage over n --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_mem_HW_usage_vs_n(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/mem_HW_usage_vs_n.png")
        fg.savefig(f"{single_plt_dir}/mem_HW_usage_vs_n.pdf", bbox_inches="tight")

        # multi
        plot_mem_HW_usage_vs_n(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- mem. HW usage over n --- #


        # --- HW util over n --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_HW_util(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/HW_util_vs_n.png")
        fg.savefig(f"{single_plt_dir}/HW_util_vs_n.pdf", bbox_inches="tight")

        # multi
        plot_HW_util(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- HW util over n --- #

        if data["running"] == False:
            # --- HW util over n --- #
            # single
            fg, ax = plt.subplots(layout="constrained")
            plot_building_batch_frac(ax, args, data, title=False)
            
            fg.savefig(f"{single_plt_dir}/building_batch_frac_vs_n.png")
            fg.savefig(f"{single_plt_dir}/building_batch_frac_vs_n.pdf", bbox_inches="tight")

            # multi
            plot_building_batch_frac(axes[ax_ctr], args, data)
            ax_ctr += 1    
            # --- HW util over n --- #            

        # --- roofline --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_HW_util(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/rooflines.png")
        fg.savefig(f"{single_plt_dir}/rooflines.pdf", bbox_inches="tight")

        # multi
        plot_HW_util(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- roofline --- #    

    if args.julia:
        # --- rel. err over order --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_RE_vs_order(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/RE_vs_order.png")
        fg.savefig(f"{single_plt_dir}/RE_vs_order.pdf", bbox_inches="tight")

        # multi
        plot_RE_vs_order(axes[ax_ctr], args, data)
        ax_ctr += 1    
        plot_RE_vs_order(axes0[ax_ctr0], args, data)
        ax_ctr0 += 1            
        # --- rel. err over order --- #        

        # --- runtimes [s] over n --- #
        # single
        fg, ax = plt.subplots(layout="constrained")
        plot_runtime_vs_n(ax, args, data, title=False)
        
        fg.savefig(f"{single_plt_dir}/runtimes_vs_n.png")
        fg.savefig(f"{single_plt_dir}/runtimes_vs_n.pdf", bbox_inches="tight")

        # multi
        plot_runtime_vs_n(axes[ax_ctr], args, data)
        ax_ctr += 1    
        # --- runtimes [s] over n --- #


    # delete remaining axes
    for i in range(ax_ctr, max_plots):
        fig.delaxes(axes[i])

    # ----------------------- plots ----------------------- #

    # finish multi-plots
    tmp = "PT" if data["PT"] else ""
    fig.suptitle(f"{tmp} moment estimation of state '{args.state}', num_meas={data['num_meas']:.3e}, var_offset={data['var_offset']}, num_samples={data['num_runs']}, m={data['orders'][-1]}")


    fig.savefig(f"{multi_plt_dir}/{args.state}.png")
    fig.savefig(f"{multi_plt_dir}/{args.state}.pdf", bbox_inches="tight")


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
