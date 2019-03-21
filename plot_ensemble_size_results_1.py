# Plots the results produced by run_ensemble_size_tests.py (Figures 14 and 15 in 
# the paper).

from pylab import *
import pickle
from string import ascii_lowercase
from pysteps.verification import ensscores, probscores

#domain = "fmi"
domain = "mch"
linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
linestyles = ['-', '-', '-', '-', '-']
#markers = ['o', 's', 'd', '^']
#markers = ['o', 'o', 'o', 'o', 'o']
markers = [None, None, None, None, None]
#markers = ["circle", "cross", "diamond", "square", "triangle-down", "triangle-up"]
minleadtime = 5
maxleadtime = 180
R_thrs = [0.1, 5.0]
upscale_factor = 1

infn = "ensemble_size_results_%s" % domain
if upscale_factor > 1:
    infn += "_%d" % upscale_factor
with open(infn + ".dat", "rb") as f:
    results = pickle.load(f)

fig_roc = figure(figsize=(4.75, 6.75))

for thri, R_thr in enumerate(R_thrs):
    if R_thr == 0.1:
        ROC_ylim = (0.77, 1.0)
    elif R_thr == 5.0:
        ROC_ylim = (0.59, 1.0)
    else:
        ROC_ylim = (0.5, 1.0)

    ax_roc = fig_roc.add_subplot(210 + thri + 1)

    ETS = dict([(es, []) for es in sorted(results.keys())])
    MAE = dict([(es, []) for es in sorted(results.keys())])
    ME = dict([(es, []) for es in sorted(results.keys())])

    ROC_areas = dict([(es, []) for es in sorted(results.keys())])
    OP = dict([(es, []) for es in sorted(results.keys())])
    CRPS = dict([(es, []) for es in sorted(results.keys())])

    for es in sorted(results.keys()):
      for lt in sorted(results[es]["CRPS"].keys()):
          d = results[es]["cat"][R_thr][lt]
          H_r = (d["H"] + d["M"]) * (d["H"] + d["F"]) / (d["H"] + d["F"] + d["M"] + d["R"])
          ETS_ = 1.0*(d["H"] - H_r) / (d["H"] + d["M"] + d["F"] - H_r)
          ETS[es].append(ETS_)
          d = results[es]["MAE"][R_thr][lt]
          MAE[es].append(1.0*d["sum"] / d["n"])
          d = results[es]["ME"][R_thr][lt]
          ME[es].append(1.0*d["sum"] / d["n"])

      for lt in sorted(results[es]["CRPS"].keys()):
          CRPS[es].append(probscores.CRPS_compute(results[es]["CRPS"][lt]))

      for lt in sorted(results[es]["ROC"][R_thr].keys()):
          a = probscores.ROC_curve_compute(results[es]["ROC"][R_thr][lt], compute_area=True)[2]
          ROC_areas[es].append(a)
      for lt in sorted(results[es]["rankhist"][R_thr].keys()):
          rh = ensscores.rankhist_compute(results[es]["rankhist"][R_thr][lt])
          OP_ = (rh[0] + rh[-1]) / sum(rh)
          OP[es].append(OP_)

    for i in range(3):
        fig = figure(figsize=(5, 3.5))
        ax = fig.gca()

        for j,es in enumerate(sorted(results.keys())):
            if i == 0:
                values = ETS[es]
            elif i == 1:
                values = MAE[es]
            else:
                values = ME[es]

            leadtimes = (np.array(sorted(results[es]["MAE"][R_thr].keys())) + 1) * 5
            ax.plot(leadtimes, values, ls=linestyles[j], marker=markers[j], lw=2,
                    ms=6, label="n=%d" % es)

        ax.set_xlim(minleadtime, maxleadtime)
        ax.set_xlabel("Lead time (minutes)")
        ax.grid(True)
        ax.legend(fontsize=12, framealpha=1.0)

        if i == 0:
            ax.set_ylabel("ETS (mm/h)")
            outfn = "ensemble_size_ETS_%s" % domain
        elif i == 1:
            ax.set_ylabel("MAE (mm/h)")
            outfn = "ensemble_size_MAE_%s" % domain
        else:
            ax.set_ylabel("ME (mm/h)")
            outfn = "ensemble_size_ME_%s" % domain

        if upscale_factor > 1:
            outfn += "_%d" % upscale_factor
        outfn += "_%.1f" % R_thr
        fig.savefig(outfn + ".pdf", bbox_inches="tight")

    fig = figure(figsize=(5, 3.5))
    ax = fig.gca()

    for i,es in enumerate(sorted(results.keys())):
      leadtimes = (np.array(sorted(results[es]["CRPS"].keys())) + 1) * 5
      ax.plot(leadtimes, CRPS[es], ls=linestyles[i], marker=markers[i], lw=2, ms=6,
              label="n=%d" % es)

    ax.set_xlim(minleadtime, maxleadtime)
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("CRPS (mm/h)")
    ax.grid(True)
    ax.legend(fontsize=12, framealpha=1.0)

    outfn = "ensemble_size_CRPS_%s" % domain
    if upscale_factor > 1:
        outfn += "_%d" % upscale_factor
    outfn += "_%.1f" % R_thr
    fig.savefig(outfn + ".pdf", bbox_inches="tight")

    #fig = figure(figsize=(5, 3.5))
    #ax = fig.gca()

    for i,es in enumerate(sorted(ROC_areas.keys())):
        leadtimes = (arange(len(ROC_areas[es])) + 1) * 5
        ax_roc.plot(leadtimes, ROC_areas[es], ls=linestyles[i], marker=markers[i],
                    color=linecolors[i], label="n=%d" % es, lw=2, ms=6)
    xt = np.hstack([5, np.arange(20, maxleadtime+20, 20)])
    ax_roc.set_xticks(xt)
    ax_roc.set_xlim(minleadtime, maxleadtime)
    ax_roc.set_ylim(ROC_ylim[0], ROC_ylim[1])
    ax_roc.grid(True)
    ax_roc.legend(fontsize=11, ncol=2, framealpha=1.0)
    ax_roc.set_title(ascii_lowercase[thri] + ')', loc="left", fontsize=12)
    if thri == len(R_thrs) - 1:
        ax_roc.set_xlabel("Lead time (minutes)", fontsize=12)
    ax_roc.set_ylabel("ROC area", fontsize=12)

#    outfn = "ensemble_size_ROC_areas_%s" % domain
#    if upscale_factor > 1:
#        outfn += "_%d" % upscale_factor
#    outfn += "_%.1f" % R_thr
#    fig.savefig(outfn + ".pdf", bbox_inches="tight")

    fig = figure(figsize=(5, 3.5))
    ax = fig.gca()
    for i,es in enumerate(sorted(OP.keys())):
        leadtimes = (arange(len(OP[es])) + 1) * 5
        ax.plot(leadtimes, OP[es], ls=linestyles[i], marker=markers[i], 
                color=linecolors[i], label="n=%d" % es, lw=2, ms=6)
    xt = np.hstack([5, np.arange(20, maxleadtime+20, 20)])
    ax.set_xticks(xt)
    ax.set_xlim(minleadtime, maxleadtime)
    ax.set_ylim(0.0, 0.47)
    ax.grid(True)
    ax.legend(fontsize=12, framealpha=1.0, loc=(1.02, 0.3))
    ax.set_xlabel("Lead time (minutes)", fontsize=12)
    ax.set_ylabel("Percentage of outliers", fontsize=12)

    outfn = "ensemble_size_OP_%s" % domain
    if upscale_factor > 1:
        outfn += "_%d" % upscale_factor
    outfn += "_%.1f" % R_thr
    fig.savefig(outfn + ".pdf", bbox_inches="tight")

outfn = "ensemble_size_ROC_areas_%s" % domain
if upscale_factor > 1:
    outfn += "_%d" % upscale_factor
fig_roc.savefig(outfn + ".pdf", bbox_inches="tight")

