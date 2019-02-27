# Plots the results produced by run_ensemble_size_tests.py (Figures 16 and 17 in 
# the paper).

from pylab import *
import pickle
from pysteps.verification import ensscores, probscores

linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
linestyles = ['-', '-', '-', '-', '-']
#markers = ['o', 's', 'd', '^']
markers = ['o', 'o', 'o', 'o', 'o']
#markers = ["circle", "cross", "diamond", "square", "triangle-down", "triangle-up"]
minleadtime = 5
maxleadtime = 180
R_thr = 0.1

with open("ensemble_size_results.dat", "rb") as f:
    results = pickle.load(f)

ROC_areas = dict([(es, []) for es in sorted(results.keys())])
OPs = dict([(es, []) for es in sorted(results.keys())])

for es in sorted(results.keys()):
  if not es in ROC_areas.keys():
      ROC_areas[es] = []
  for lt in sorted(results[es]["ROC"][R_thr].keys()):
      a = probscores.ROC_curve_compute(results[es]["ROC"][R_thr][lt], compute_area=True)[2]
      ROC_areas[es].append(a)
  for lt in sorted(results[es]["rankhist"][R_thr].keys()):
      rh = ensscores.rankhist_compute(results[es]["rankhist"][R_thr][lt])
      OP = (rh[0] + rh[-1]) / sum(rh)
      OPs[es].append(OP)

fig = figure(figsize=(5, 3.5))
ax = fig.gca()
for i,es in enumerate(sorted(ROC_areas.keys())):
    leadtimes = (arange(len(ROC_areas[es])) + 1) * 5
    ax.plot(leadtimes, ROC_areas[es], ls=linestyles[i], marker=markers[i], 
            color=linecolors[i], label="n=%d" % es, lw=2, ms=6)
xt = np.hstack([5, np.arange(20, maxleadtime+20, 20)])
ax.set_xticks(xt)
ax.set_xlim(minleadtime, maxleadtime)
ax.set_ylim(0.7, 1.0)
ax.grid(True)
ax.legend(fontsize=12, framealpha=1.0)
ax.set_xlabel("Lead time (minutes)", fontsize=12)
ax.set_ylabel("ROC area", fontsize=12)

fig.savefig("ensemble_size_ROC_areas.pdf", bbox_inches="tight")

fig = figure(figsize=(5, 3.5))
ax = fig.gca()
for i,es in enumerate(sorted(OPs.keys())):
    leadtimes = (arange(len(OPs[es])) + 1) * 5
    ax.plot(leadtimes, OPs[es], ls=linestyles[i], marker=markers[i], 
            color=linecolors[i], label="n=%d" % es, lw=2, ms=6)
xt = np.hstack([5, np.arange(20, maxleadtime+20, 20)])
ax.set_xticks(xt)
ax.set_xlim(minleadtime, maxleadtime)
ax.set_ylim(0.0, 0.4)
ax.grid(True)
ax.legend(fontsize=12, framealpha=1.0, loc=(1.02, 0.3))
ax.set_xlabel("Lead time (minutes)", fontsize=12)
ax.set_ylabel("Percentage of outliers", fontsize=12)

fig.savefig("ensemble_size_OPs.pdf", bbox_inches="tight")
