# Plots the results produced by run_parallel_scaling_tests.py.

import pickle
from matplotlib import pyplot
import numpy as np

# the maximum number of threads to include in the plot
max_num_threads = 8

linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

results = {}
results["fmi"] = pickle.load(open("parallel_scaling_results_fmi.dat","rb"))
results["mch"] = pickle.load(open("parallel_scaling_results_mch.dat","rb"))

fig = pyplot.figure(figsize=(4.3, 6.5))
ax = fig.subplots(nrows=2, ncols=1)

y_max = {"fmi": 0.0, "mch": 0.0}

for i, d in enumerate(["mch", "fmi"]):
    for j, es in enumerate(sorted(results[d].keys())):
        #if d == "fmi" and es == 48:
        #    continue

        nw = sorted(results[d][es].keys())
        t = [results[d][es][nw][1]/60.0 for nw in nw]
        ax[i].plot(nw, t, color=linecolors[j], ls="-", lw=2, marker='D',
                   label="n=%d" % es)

        # Plotting the initialization time is commented out because it's
        # negligibly small.

        # Take the first result, because the STEPS initialization time does not
        # depend on the number of threads.
        #max_ens_size = sorted(results[d].keys())[0]
        #ax[i].plot([0, max_num_threads], [results[d][max_ens_size][1][0]/60.0]*2,
        #           "k--", lw=2)

        #ax[i].text(max_num_threads/2-1, results[d][es][1][0]/60.0*1.5,
        #           "Initialization time", fontsize=10)

        ax[i].legend(fontsize=12)
        ax[i].grid(True)
        ax[i].set_xlim(1, max_num_threads)

        y_max[d] = max(y_max[d], np.max(t))

fig.text(0.02, 0.5, "Computation time (minutes)", va="center", ha="center",
         rotation='vertical', fontsize=12)

ax[0].set_yticks(np.arange(0, 1.1*y_max["mch"], 1))
ax[0].tick_params(labelsize=10)
ax[1].set_xticks(nw, ["%d" % nw_ for nw_ in nw])
ax[1].set_yticks(np.arange(0, 1.1*y_max["fmi"], 2))
ax[1].tick_params(labelsize=10)

ax[1].set_xlabel("Number of threads", fontsize=12)

ax[0].set_title("(a) MeteoSwiss", fontsize=12)
ax[1].set_title("(b) FMI", fontsize=12)

pyplot.savefig("ensemble_comp_times.pdf", bbox_inches="tight")
