# Plots the results produced by run_optflow comparison.py (Figure 11 in the 
# paper).

from pylab import *
import pickle

# line and marker parameters
#linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
#              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#ff7f0e", "#2ca02c"]
linestyles = ['-', '-', '-', '-', '-', '-']
markers = ['o', 'o', 'o', 's', 's', 's']
# legend labels corresponding to the optical flow methods
legend_labels = {"darts":"DARTS", "lucaskanade":"Lucas-Kanade", "vet":"VET"}
# minimum lead time to plot (minutes)
minleadtime = 5
# maximum lead time to plot (minutes)
maxleadtime = 120
# nowcast methods to include in the plot
nowcast_methods = ["advection", "sprog"]

results = {}
for ncm in nowcast_methods:
    with open("optflow_eval_results_%s.dat" % ncm, "rb") as f:
        results[ncm] = pickle.load(f)

fig_csi = figure()
ax_csi = fig_csi.gca()
fig_mae = figure()
ax_mae = fig_mae.gca()

i = 0
for ncm in nowcast_methods:
    oflow_methods = sorted(results[ncm].keys())

    for oflow_method in oflow_methods:
        n_samples = array(results[ncm][oflow_method]["n_samples"])
        ncm_ = "Advection" if ncm == "advection" else "S-PROG"
        lbl = ncm_ + ' / ' + legend_labels[oflow_method]
        csi = array(results[ncm][oflow_method]["CSI"]) / n_samples
        leadtimes = (arange(len(csi)) + 1) * 5
        ax_csi.plot(leadtimes, csi, color=linecolors[i], ls=linestyles[i], 
                    marker=markers[i], label=lbl, lw=2, ms=6)
        mae = array(results[ncm][oflow_method]["MAE"]) / n_samples
        ax_mae.plot(leadtimes, mae, color=linecolors[i], ls=linestyles[i], 
                    marker=markers[i], label=lbl, lw=2, ms=6)

        i += 1

ax_csi.legend(loc=1, fontsize=12)
ax_csi.set_xlabel("Lead time (minutes)", fontsize=12)
ax_csi.set_ylabel("CSI", fontsize=12)
ax_csi.grid(True)
ax_csi.set_xlim(minleadtime, maxleadtime)
ax_csi.set_ylim(0.35, 0.9)

ax_mae.legend(loc=4, fontsize=12)
ax_mae.set_xlabel("Lead time (minutes)", fontsize=12)
ax_mae.set_ylabel("MAE", fontsize=12)
ax_mae.grid(True)
ax_mae.set_xlim(minleadtime, maxleadtime)
ax_mae.set_ylim(0.4, 4.2)

fig_csi.savefig("oflow_benchmark_csi.pdf",  bbox_inches="tight")
fig_mae.savefig("oflow_benchmark_mae.pdf", bbox_inches="tight")
