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


def cmp_ERR_vs_order(ax, args, data1, data2, title=True):
    for idx, n in enumerate(data1["qubits"]):
        c = data1["colors"][idx%len(data1["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]
        ax.plot(data1["orders"], data2["ERR"][idx]/data1["ERR"][idx], c=c, marker=marker, label=f"n={n}")
    ax.set_xlabel("order")
    ax.set_ylabel("ERR2/ERR1")
    if title:
        ax.set_title(f"Fraction of median abs. error (ERR) of 2 to 1\nof moment estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)    

def cmp_RE_frac_vs_order(ax, args, data1, data2, title=True):
    for idx, n in enumerate(data1["qubits"]):
        c = data1["colors"][idx%len(data1["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]        
        ax.semilogy(data1["orders"], data2["RE"][idx]/data1["RE"][idx], c=c, marker=marker, label=f"n={n}")
    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75)
    ax.set_xlabel("order")
    ax.set_ylabel("RE2/RE1")
    if title:
        ax.set_title(f"Fraction of median rel. error (RE) of 2 to 1\nof moment estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_frac_to_plot = 1e2
    max_frac = np.max(data2["RE"][idx]/data1["RE"][idx])
    if max_frac > max_frac_to_plot:
        ax.set_ylim(ymin=None, ymax=max_frac_to_plot)       

def cmp_median_RE_frac_vs_n(ax, args, data1, data2, in_data1, in_data2, title=True):
    num_runs = data1["num_runs"]
    len_orders = len(data1["orders"])

    avg_RE_fracs = np.zeros([len(data1["qubits"]), len(data1["orders"])])

    for idx, n in enumerate(data1["qubits"]):
        skip_o1 = 1 # we intentionally skip order 1 here
        moments = np.tile(in_data1[n]["moments"][skip_o1:], num_runs).reshape(num_runs, len_orders)
        est_moments = in_data1[n]["est_moments"].reshape(num_runs, len_orders + 1)
        errs = np.abs(moments - est_moments[:, skip_o1:])
        errs[errs == 0] = np.nan

        rel_errs1 = errs/np.abs(moments)

        moments = np.tile(in_data2[n]["moments"][skip_o1:], num_runs).reshape(num_runs, len_orders)
        est_moments = in_data2[n]["est_moments"].reshape(num_runs, len_orders + 1)
        errs = np.abs(moments - est_moments[:, skip_o1:])
        errs[errs == 0] = np.nan        

        rel_errs2 = errs/np.abs(moments)

        avg_RE_fracs[idx] = np.mean(rel_errs2 / rel_errs1, axis=0)
    avg_RE_fracs = np.median(avg_RE_fracs, axis=1)
    
    idx = 0
    c = data1["colors"][idx%len(data1["colors"])]
    marker = data1["markers"][idx%len(data1["markers"])]        
    ax.plot(data1["qubits"], avg_RE_fracs, c=c, marker=marker)
    ax.axhline(y=1, linestyle="-.", c="grey", alpha=0.75)
    ax.axhline(y=2**(1/2), linestyle="--", c="black", alpha=0.75, label=r"$\sqrt{2}$")
    ax.axhline(y=np.e**(1/2), linestyle="--", c="grey", alpha=0.75, label=r"$\sqrt{e}$")
    ax.set_xlabel("n")
    ax.set_ylabel("RE_2/RE_1")
    if title:
        ax.set_title(f"Improvement in accuracy of running to batched algoritm\naverage over {num_runs} runs, median over orders up to {data1["orders"][-1]}")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    # max_frac_to_plot = 1e2
    # max_frac = np.max(data2["RE"][idx]/data1["RE"][idx])
    # if max_frac > max_frac_to_plot:
    #     ax.set_ylim(ymin=None, ymax=max_frac_to_plot)            

def cmp_avg_RE_frac_n_vs_order(ax, args, data1, data2, in_data1, in_data2, title=True):
    num_runs = data1["num_runs"]
    len_orders = len(data1["orders"])

    avg_RE_fracs = np.zeros([len(data1["qubits"]), len(data1["orders"])])

    for idx, n in enumerate(data1["qubits"]):
        skip_o1 = 1 # we intentionally skip order 1 here
        moments = np.tile(in_data1[n]["moments"][skip_o1:], num_runs).reshape(num_runs, len_orders)
        est_moments = in_data1[n]["est_moments"].reshape(num_runs, len_orders + 1)
        errs = np.abs(moments - est_moments[:, skip_o1:])
        errs[errs == 0] = np.nan

        rel_errs1 = errs/np.abs(moments)

        moments = np.tile(in_data2[n]["moments"][skip_o1:], num_runs).reshape(num_runs, len_orders)
        est_moments = in_data2[n]["est_moments"].reshape(num_runs, len_orders + 1)
        errs = np.abs(moments - est_moments[:, skip_o1:])
        errs[errs == 0] = np.nan        

        rel_errs2 = errs/np.abs(moments)

        avg_RE_fracs[idx] = np.mean(rel_errs2 / rel_errs1, axis=0)
    
    for idx, n in enumerate(data1["qubits"]):
        c = data1["colors"][idx%len(data1["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]        
        ax.semilogy(data1["orders"], avg_RE_fracs[idx], c=c, marker=marker, label=f"n = {n}")
    ax.axhline(y=1, linestyle="-.", c="grey", alpha=0.75)
    ax.axhline(y=2**(1/2), linestyle="--", c="black", alpha=0.75, label=r"$\sqrt{2}$")
    ax.axhline(y=np.e**(1/2), linestyle="--", c="grey", alpha=0.75, label=r"$\sqrt{e}$")
    ax.set_xlabel("order")
    ax.set_ylabel("avg RE2/RE1")
    if title:
        ax.set_title(f"Avg. (over n) Fraction of median rel. error (RE) of 2 to 1\nof moment estimation over order")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    # max_frac_to_plot = 1e2
    # max_frac = np.max(data2["RE"][idx]/data1["RE"][idx])
    # if max_frac > max_frac_to_plot:
    #     ax.set_ylim(ymin=None, ymax=max_frac_to_plot)       

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

def cmp_est_rel_err_vs_order(ax, args, data1, data2, title=True):
    for idx, n in enumerate(data1["qubits"]):
        c1 = data1["colors"][idx%len(data1["colors"])]
        c2 = data2["colors"][idx%len(data2["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]   

        ax.semilogy(data2["orders"], data2["ERE"][idx], c=c2, marker=marker, label=f"n={n}, 2")
        ax.semilogy(data1["orders"], data1["ERE"][idx], c=c1, marker=marker, label=f"n={n}, 1")

    ax.axhline(y=args.tol, linestyle="--", c="grey", alpha=0.75, label="tol")
    ax.set_xlabel("order")
    ax.set_ylabel("estimated rel. err.")
    if title:
        ax.set_title(f"Median estimated upper bound of RE obtained\nthrough variance estimation over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

    max_ERE_to_plot = 1e2
    max_ERE = max(np.max(data1["ERE"]), np.max(data1["ERE"]))
    if max_ERE > max_ERE_to_plot:
        ax.set_ylim(ymin=None, ymax=max_ERE_to_plot)   

def cmp_CIOE_vs_order(ax, args, data1, data2, title=True):
    for idx, n in enumerate(data1["qubits"]):
        c1 = data1["colors"][idx%len(data1["colors"])]
        c2 = data2["colors"][idx%len(data2["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]     

        ax.semilogy(data2["orders"], data2["CIOE"][idx], c=c2, marker=marker, label=f"n={n}, 2")
        ax.semilogy(data1["orders"], data1["CIOE"][idx], c=c1, marker=marker, label=f"n={n}, 1")
        if data1["num_runs"] > 1:
            ax.semilogy(data2["orders"], data2["CIOE_min"][idx], linestyle="--", linewidth=data2["lw"], c=c2)
            ax.semilogy(data1["orders"], data1["CIOE_min"][idx], linestyle="--", linewidth=data1["lw"], c=c1)

    ax.axhline(y=1, linestyle="--", c="grey", alpha=0.75)
    ax.set_xlabel("order")
    ax.set_ylabel("Overestimation (CI/ERR)")
    if title:
        ax.set_title(f"Median overestimation of {args.CI_perc}% conf. interval (CI) to the true error\n(ERR) of moment estimation over order for different n\nsolid lines ... median, dashed lines ... min")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)


def cmp_corr_est_moments_vs_order(ax, args, data1, data2, positives=True, negatives=False, title=True):
    assert (positives or negatives) and not (positives and negatives)
    title_str = "positive" if positives else "abs. value of negative"
    for idx, n in enumerate(data1["qubits"]):
        c1 = data1["colors"][idx%len(data1["colors"])]
        c2 = data2["colors"][idx%len(data2["colors"])]
        marker = data1["markers"][idx%len(data1["markers"])]          
        moments1 = data1["moments"][idx]
        moments2 = data2["moments"][idx]

        pos_neg_mask1 = (moments1 >= 0) if positives else (moments1 <= 0)
        pos_neg_mask2 = (moments2 >= 0) if positives else (moments2 <= 0)

        converged1 = (data1["ERE"][idx] < args.tol)
        converged2 = (data2["ERE"][idx] < args.tol)

        mask1 = pos_neg_mask1 & converged1
        mask2 = pos_neg_mask2 & converged2

        non_converged1 = ~converged1
        non_converged2 = ~converged2

        ax.plot(data2["orders"][mask2], np.abs(moments2[mask2]), c=c2, lw=1, label=f"n={n}, 2")
        ax.plot(data1["orders"][mask1], np.abs(moments1[mask1]), c=c1, lw=1, label=f"n={n}, 1")

        point_est1 = np.abs(data1["est_moments"][idx][mask1])
        point_est2 = np.abs(data2["est_moments"][idx][mask2])

        yerr1 = data1["CI"][idx][mask1]
        yerr2 = data2["CI"][idx][mask2]

        ax.errorbar(data2["orders"][mask2], 
                    point_est2, 
                    yerr=yerr2,
                    fmt=marker,
                    capsize=4,
                    capthick=1.2,
                    linestyle="None",
                    markersize=4,
                    color=c2,
                    ecolor=c2
                    )    
        ax.errorbar(data1["orders"][mask1], 
                    point_est1, 
                    yerr=yerr1,
                    fmt=marker,
                    capsize=4,
                    capthick=1.2,
                    linestyle="None",
                    markersize=4,
                    color=c1,
                    ecolor=c1
                    )
    
    ax.set_yscale('log')

    ax.set_xlabel("order")
    ax.set_ylabel("moment")
    if title:
        ax.set_title(f"Converged {title_str} estimated (dots; median; tol={args.tol})\nwith error bars and true moments (lines) over order for different n")
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)

def cmp_SU_vs_n(ax, args, data1, data2, title=True):
    if not np.all(data1["num_threads"] == data2["num_threads"]) and data1["running"] == data2["running"]:
        ax.semilogy(data1["qubits"], data2["runtimes"]/data1["runtimes"] , marker="v", label="speedup")
        ax.semilogy(data1["qubits"], 2**(data1["qubits"]) , linestyle="--", lw=data1["lw"], label="ideal speedup")
        ax.set_ylim(ymin=0.8*np.min(data2["runtimes"]/data1["runtimes"]), ymax=1.2*np.max(data2["runtimes"]/data1["runtimes"])) 
        ax.set_yticks([4, 8, 16, 32, 64])
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())     
    else:
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

def cmp_runtime_vs_n(ax, args, data1, data2, title=True):
    ax.semilogy(data1["qubits"], data1["runtimes"] , marker="v", label="1")
    ax.semilogy(data2["qubits"], data2["runtimes"] , marker="s", label="2")
    ax.set_xlabel("n")
    ax.set_ylabel("runtime [s]")
    if title:
        ax.set_title(f"median runtime for estimating all moments\nup to order {data1['orders'][-1]} over n")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)    

def cmp_comp_HW_usage_vs_n(ax, args, data1, data2, title=True):
    cs1 = data1["colors"]
    cs2 = data2["colors"]

    ax.semilogy(data2["qubits"], data2["vector_width"], c=cs2[0], marker="o", label="vector-width, 2")
    ax.semilogy(data2["qubits"], data2["cores"], c=cs2[1], marker="s", label="#cores, 2")
    ax.semilogy(data2["qubits"], data2["num_threads"], c=cs2[2], marker="v", label="#threads, 2")
    ax.semilogy(data2["qubits"], data2["num_ranks"], c=cs2[2], marker="x", label="#nodes, 2")

    ax.semilogy(data1["qubits"], data1["vector_width"], c=cs1[0], marker="o", label="vector-width, 1")
    ax.semilogy(data1["qubits"], data1["cores"], c=cs1[1], marker="s", label="#cores, 1")
    ax.semilogy(data1["qubits"], data1["num_threads"], c=cs1[2], marker="v", label="#threads, 1")
    ax.semilogy(data1["qubits"], data1["num_ranks"], c=cs1[2], marker="x", label="#nodes, 1")

    ax.set_xlabel("n")
    ax.set_ylabel("#cores/ #threads/ #nodes/ vector-width [B]")
    if title:
        ax.set_title(f"Used number of threads, number of cores,\nand vector-widths in bytes over num_qubits")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)

def cmp_mem_HW_usage_vs_n(ax, args, data1, data2, title=True):
    cs1 = data1["colors"]
    cs2 = data2["colors"]


    ax.semilogy(data2["qubits"], data2["total_mem"], c=cs2[0], marker="v", label="Memory, 2")
    ax.semilogy(data2["qubits"], data2["pk_per_core"], c=cs2[1], marker="v", label="p_k per core, 2")
    ax.semilogy(data2["qubits"], data2["xy_per_core"], c=cs2[2], marker="o", label="x, y per core, 2")

    ax.semilogy(data1["qubits"], data1["total_mem"], c=cs1[0], marker="v", label="Memory, 1")
    ax.semilogy(data1["qubits"], data1["pk_per_core"], c=cs1[1], marker="v", label="p_k per core, 1")
    ax.semilogy(data1["qubits"],  data1["xy_per_core"], c=cs1[2], marker="o", label="x, y per core, 1")

    ax.axhline(y=data1["HW"]["L1d"], linestyle="--", c="slategrey", alpha=0.75, label=f"L1d per core")
    ax.axhline(y=data1["HW"]["L2"], linestyle="--", c="dimgrey", alpha=0.75, label=f"L2 per core")

    ax.set_xlabel("n")
    ax.set_ylabel("MB")
    if title:
        ax.set_title(f"Total required memory (meas. data + matrices)\nwith HW ({data1['HW']['cpu_name']}) metrics")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)  

def cmp_HW_util(ax, args, data1, data2, title=True):
    cs1 = data1["colors"]
    cs2 = data2["colors"]

    ax.plot(data2["qubits"], 100 * data2["AP"]/data2["PP"], c=cs2[0], marker="v", label="abs. HW util, 2")
    ax.plot(data1["qubits"], 100 * data1["AP"]/data1["PP"], c=cs1[0], marker="v", label="abs. HW util, 1")

    achievable_PP1 = compute_PP_vs_n(data1, args)
    achievable_PP2 = compute_PP_vs_n(data2, args)
    ax.plot(data2["qubits"], 100 * data2["AP"]/achievable_PP2, c=cs2[1], marker="s", label="rel. HW util, 2")
    ax.plot(data1["qubits"], 100 * data1["AP"]/achievable_PP1, c=cs1[1], marker="s", label="rel. HW util, 1")

    ax.axhline(y=100, linestyle="--", c="black", alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("HW utilization [%]")
    if title:
        ax.set_title(f"Hardware-utilization over n with HW {data1['HW']['cpu_name']}\nPP = (#cores) * (base_freq) * (data_path_width)\n/ (float_size) * (#fma_pipes) * 2 (fma)")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)    
    ax.legend(ncol=2, fontsize=8, markerscale=0.6, handlelength=1.0)  

def cmp_roofline_model(ax, args, data1, data2, title=True):
    fac = 1e3 # GFLOPs (M to G)

    AI1 = compute_AI(data1)
    PPs1 = compute_PP_vs_n(data1, args)
    BWs1 = compute_BWs(data1)

    AI2 = compute_AI(data2)
    PPs2 = compute_PP_vs_n(data2, args)
    BWs2 = compute_BWs(data2)

    plot_rooflines(ax, data1, PPs1, BWs1, fac)
    plot_rooflines(ax, data2, PPs2, BWs2, fac, plot_qb_label=False)

    scatter_roofline_data_points(ax, data1, AI1, fac)
    scatter_roofline_data_points(ax, data2, AI2, fac)

    ax.set_xlabel("Arithmetic Intensity")
    ax.set_ylabel("GFLOPs")
    if title:
        ax.set_title(f"Roofline model with computed peak perf. and measured\nDRAM bandwidths of {data1['HW']['cpu_name']} processor")
    ax.grid(which="major", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    ax.grid(which="minor", visible=False)
    ax.legend(ncol=2, fontsize=4, markerscale=0.6, handlelength=1.0)    


# ---------------------- main ----------------------
def run_all_plots(args):

    # extract data
    in_data1, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points = load_data(args, args.data_dir1)
    in_data2, qubits2, orders2, PT2, var_offset2, num_meas2, num_runs2, num_data_points2 = load_data(args, args.data_dir2)

    assert np.all(qubits == qubits2)
    assert np.all(orders == orders2)
    assert PT == PT2
    assert var_offset == var_offset2
    assert num_runs == num_runs2

    data1 = extract_data(args, in_data1, qubits, orders, PT, var_offset, num_meas, num_runs, num_data_points, compute_DICI=False)
    data2 = extract_data(args, in_data2, qubits, orders, PT, var_offset, num_meas2, num_runs, num_data_points2, compute_DICI=False)

    log_run_details(args, data1, args.data_dir1, name="info_1.txt", print_state=False)
    log_run_details(args, data2, args.data_dir2, name="info_2.txt", print_state=False)

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
    multi_plt_dir = f"{args.out_dir}/multi"
    single_plt_dir = f"{args.out_dir}/single"
    os.makedirs(multi_plt_dir, exist_ok=True)    
    os.makedirs(single_plt_dir, exist_ok=True)    

    # subplots
    rows = 3
    cols = 3
    cell_w = 6 
    cell_h = 4.5
    dpi = 140
    figsize = (cols * cell_w, rows * cell_h)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    max_plots = rows * cols
    ax_ctr = 0 

    # SU
    rows0 = 1
    cols0 = 3
    cell_w0 = 6 
    cell_h0 = 4.5
    dpi0 = 140
    figsize0 = (cols0 * cell_w0, rows0 * cell_h0)

    fig0, axes0 = plt.subplots(rows0, cols0, figsize=figsize0, dpi=dpi0)
    axes0 = axes0.flatten()
    max_plots0 = rows0 * cols0
    ax_ctr0 = 0 


    # ----------------------- plots ----------------------- #

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_RE_frac_vs_order(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.png")
    fg.savefig(f"{single_plt_dir}/RE_frac_vs_order.pdf", bbox_inches="tight")

    # multi
    cmp_RE_frac_vs_order(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    
    # --- rel. err over order --- #

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_RE_vs_order(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/RE_vs_order.png")
    fg.savefig(f"{single_plt_dir}/RE_vs_order.pdf", bbox_inches="tight")
    # --- rel. err over order --- #    

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_median_RE_frac_vs_n(ax, args, data1, data2, in_data1, in_data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/median_RE_frac_vs_n.png")
    fg.savefig(f"{single_plt_dir}/median_RE_frac_vs_n.pdf", bbox_inches="tight")
    # --- rel. err over order --- #        

    # --- rel. err over order --- #
    # single
    fg, ax = plt.subplots()
    cmp_avg_RE_frac_n_vs_order(ax, args, data1, data2, in_data1, in_data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/avg_RE_frac_n_vs_order.png")
    fg.savefig(f"{single_plt_dir}/avg_RE_frac_n_vs_order.pdf", bbox_inches="tight")
    # --- rel. err over order --- #       

    if data1["running"] or data2["running"]:
        # --- rel. uncertainty over order --- #
        # single
        fg, ax = plt.subplots()
        cmp_est_rel_err_vs_order(ax, args, data1, data2, title=False)
        fg.tight_layout()
        fg.savefig(f"{single_plt_dir}/ERE_vs_order.png")
        fg.savefig(f"{single_plt_dir}/ERE_vs_order.pdf", bbox_inches="tight")
        
        # multi
        cmp_est_rel_err_vs_order(axes[ax_ctr], args, data1, data2)
        ax_ctr += 1    
        # --- rel. uncertainty over order --- #

        # --- CI overestimation over order --- #
        # single
        fg, ax = plt.subplots()
        cmp_CIOE_vs_order(ax, args, data1, data2, title=False)
        fg.tight_layout()
        fg.savefig(f"{single_plt_dir}/CIOE_vs_order.png")
        fg.savefig(f"{single_plt_dir}/CIOE_vs_order.pdf", bbox_inches="tight")

        # multi
        cmp_CIOE_vs_order(axes[ax_ctr], args, data1, data2)
        ax_ctr += 1    
        # --- CI overestimation over order --- #

    # --- corr. moments and median est. moments over order --- #
    # single
    # positives
    fg, ax = plt.subplots()
    cmp_corr_est_moments_vs_order(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/pos_true_est_moments_vs_order.png")
    fg.savefig(f"{single_plt_dir}/pos_true_est_moments_vs_order.pdf", bbox_inches="tight")

    # negatives
    fg, ax = plt.subplots()
    cmp_corr_est_moments_vs_order(ax, args, data1, data2, positives=False, negatives=True, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/neg_true_est_moments_vs_order.png")    
    fg.savefig(f"{single_plt_dir}/neg_true_est_moments_vs_order.pdf", bbox_inches="tight")    

    # multi
    cmp_corr_est_moments_vs_order(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    
    # --- corr. moments and median est. moments over order --- #


    # --- SU over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_SU_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/SU_vs_n.png")
    fg.savefig(f"{single_plt_dir}/SU_vs_n.pdf", bbox_inches="tight")

    # multi
    cmp_SU_vs_n(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    

    # SU
    cmp_SU_vs_n(axes0[ax_ctr0], args, data1, data2)
    ax_ctr0 += 1    
    # --- SU over n --- #

    # --- runtimes [s] over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_runtime_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/runtime_vs_n.png")
    fg.savefig(f"{single_plt_dir}/runtime_vs_n.pdf", bbox_inches="tight")
    # --- runtimes [s] over n --- #    

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

    # multi
    cmp_comp_HW_usage_vs_n(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    

    # multi
    cmp_comp_HW_usage_vs_n(axes0[ax_ctr0], args, data1, data2)
    ax_ctr0 += 1        
    # --- comp. HW usage over n --- #


    # --- mem. HW usage over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_mem_HW_usage_vs_n(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/mem_HW_usage_vs_n.png")
    fg.savefig(f"{single_plt_dir}/mem_HW_usage_vs_n.pdf", bbox_inches="tight")

    # multi
    cmp_mem_HW_usage_vs_n(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    
    # --- mem. HW usage over n --- #


    # --- HW util over n --- #
    # single
    fg, ax = plt.subplots()
    cmp_HW_util(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/HW_util_vs_n.png")
    fg.savefig(f"{single_plt_dir}/HW_util_vs_n.pdf", bbox_inches="tight")

    # multi
    cmp_HW_util(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    
    # --- HW util over n --- #

    # --- roofline --- #
    # single
    fg, ax = plt.subplots()
    cmp_roofline_model(ax, args, data1, data2, title=False)
    fg.tight_layout()
    fg.savefig(f"{single_plt_dir}/rooflines.png")
    fg.savefig(f"{single_plt_dir}/rooflines.pdf", bbox_inches="tight")

    # multi
    cmp_roofline_model(axes[ax_ctr], args, data1, data2)
    ax_ctr += 1    

    # multi
    cmp_roofline_model(axes0[ax_ctr0], args, data1, data2)
    ax_ctr0 += 1        
    # --- roofline --- #    

    # finish multi-plot SU
    tmp = "PT" if data1["PT"] else ""
    fig0.suptitle(f"{tmp} moment estimation of 2 to 1, num_meas={data1['num_meas']:.3e}, var_offset={data1['var_offset']}, num_samples={data1['num_runs']}")
    fig0.tight_layout()
    fig0.savefig(f"{multi_plt_dir}/SU.png")
    fig0.savefig(f"{multi_plt_dir}/SU.pdf", bbox_inches="tight")
    

    # delete remaining axes
    for i in range(ax_ctr, max_plots):
        fig.delaxes(axes[i])

    # ----------------------- plots ----------------------- #

    # finish multi-plots
    tmp = "PT" if data1["PT"] else ""
    fig.suptitle(f"{tmp} moment estimation of 2 to 1, num_meas={data1['num_meas']:.3e}, var_offset={data1['var_offset']}, num_samples={data1['num_runs']}")

    fig.tight_layout()
    fig.savefig(f"{multi_plt_dir}/cmp.png")
    fig.savefig(f"{multi_plt_dir}/cmp.pdf", bbox_inches="tight")


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
