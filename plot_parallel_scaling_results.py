# Plots the results produced by run_parallel_scaling_tests.py.

import pickle
from matplotlib import pyplot
import numpy as np

# the maximum number of threads to include in the plot
max_num_threads = 12

linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

results = {}
results["fmi"] = pickle.load(open("parallel_scaling_results_fmi.dat","rb"))
results["mch"] = pickle.load(open("parallel_scaling_results_mch.dat","rb"))

fig = pyplot.figure(figsize=(4.3, 6.5))
ax = fig.subplots(nrows=2, ncols=1)

for i, d in enumerate(["mch", "fmi"]):
    y_min = 1e6
    y_max = 0.0

    for j, es in enumerate(sorted(results[d].keys())[1:]):
        nw_ = sorted(results[d][es].keys())
        nw = []
        for k in range(len(nw_)):
            if int(es / nw_[k]) == es / nw_[k]:
                nw.append(nw_[k])

        t = [results[d][es][nw_][1]/60.0 for nw_ in nw]
        # TODO: Consider using a logarithmic y-axis.
        ax[i].loglog(nw, t, color=linecolors[j], ls="-", lw=2, marker='D',
                       label="n=%d" % es, basey=2)

        # Plotting the initialization time is commented out because it's
        # negligibly small.

        # Take the first result, because the STEPS initialization time does not
        # depend on the number of threads.
        #max_ens_size = sorted(results[d].keys())[0]
        #ax[i].plot([0, max_num_threads], [results[d][max_ens_size][1][0]/60.0]*2,
        #           "k--", lw=2)

        #ax[i].text(max_num_threads/2-1, results[d][es][1][0]/60.0*1.5,
        #           "Initialization time", fontsize=10)

        y_min = min(y_min, np.min(t))
        y_max = max(y_max, np.max(t))

    #xts = np.arange(1, max_num_threads+1)
    xts = [1, 2, 3, 4, 6, 8, 12]
    ax[i].set_xticks(xts)
    ax[i].set_xticks([], minor=True)
    ax[i].set_xticklabels(["%d" % nw for nw in xts])

    #yts = np.logspace(-2, np.log2(1.1*y_max), 7, base=2.0)
    yts = [2.0**k for k in np.arange(np.floor(np.log2(0.9*y_min)), 
           np.ceil(np.log2(1.1*y_max)))]
    ax[i].set_yticks(yts)
    ax[i].set_yticklabels(["%.1f" % v if v != int(v) else "%d" % v for v in yts])
    ax[i].set_yticks([], minor=True)
    ax[i].tick_params(labelsize=10)

    ax[i].legend(fontsize=12, loc=3, framealpha=1.0)
    ax[i].grid(True)
    ax[i].set_xlim(1, max_num_threads)

fig.text(0.01, 0.5, "Computation time (minutes)", va="center", ha="center",
         rotation='vertical', fontsize=12)

#ax[0].set_xticks(nw, ["%d" % nw_ for nw_ in nw])
#ax[0].set_yticks(np.arange(0, 1.1*y_max["mch"], 1))
#ax[1].set_xticks(nw, ["%d" % nw_ for nw_ in nw])
#ax[1].set_yticks(np.arange(0, 1.1*y_max["fmi"], 2))

ax[1].set_xlabel("Number of threads", fontsize=12)

ax[0].set_title("a)", loc="left", fontsize=12)
ax[1].set_title("b)", loc="left", fontsize=12)

pyplot.savefig("ensemble_comp_times.pdf", bbox_inches="tight")
