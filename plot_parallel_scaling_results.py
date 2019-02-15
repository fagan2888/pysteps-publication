# Plots the results produced by run_parallel_scaling_tests.py.

import pickle
from matplotlib import pyplot

# the domain: "fmi" or "mch"
domain = "mch"
# the maximum number of threads to include in the plot
max_num_threads = 8

linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

results = pickle.load(open("parallel_scaling_results_%s.dat" % domain,"rb"))

pyplot.figure(figsize=(5, 3.75))

# Take the first result, because the STEPS initialization time does not
# depend on the number of threads.
max_ens_size = sorted(results.keys())[0]
pyplot.plot([0, max_num_threads], [results[max_ens_size][1][0]/60.0]*2, "k--", lw=2)

for i,es in enumerate(sorted(results.keys())):
    nw = sorted(results[es].keys())
    t = [results[es][nw][1]/60.0 for nw in nw]
    pyplot.plot(nw, t, color=linecolors[i], ls="-", lw=2, marker='D',
                label="n=%d" % es)

pyplot.text(max_num_threads/2-1, results[es][1][0]/60.0*1.5,
            "Initialization time", fontsize=10)

pyplot.legend(fontsize=12)
pyplot.xticks(nw, ["%d" % nw_ for nw_ in nw])
pyplot.xlabel("Number of threads", fontsize=12)
pyplot.ylabel("Computation time (minutes)", fontsize=12)
pyplot.grid(True)
pyplot.xlim(1, max_num_threads)

pyplot.savefig("ensemble_comp_times_%s.pdf" % domain, bbox_inches="tight")
