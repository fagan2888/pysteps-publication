# Plots the results produced by run_ensemble_size_tests.py (Figures 9 and 10 in 
# the paper).

from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
from pysteps.verification import ensscores, probscores

#domain = "fmi"
domain = "mch"
linecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
#leadtimes = [15, 30, 60, 90]
leadtimes = [2, 11, 17, 23]
ensemble_size = 24

with open("ensemble_size_results_%s.dat" % domain, "rb") as f:
    results = pickle.load(f)

for R_thr in results[ensemble_size]["ROC"].keys():
    figure(figsize=(5, 3.5))

    plot([0, 1], [0, 1], "k--")

    for i,lt in enumerate(leadtimes):
        ROC = results[ensemble_size]["ROC"][R_thr][lt]
        POFD,POD,area = probscores.ROC_curve_compute(ROC, compute_area=True)
        plot(POFD, POD, color=linecolors[i], linestyle='-', marker='D', 
             label="%d minutes (area=%.2f)" % (((lt+1)*5), area))

    xlim(0, 1)
    ylim(0, 1)
    xlabel("False alarm rate (POFD)")
    ylabel("Probability of detection (POD)")
    grid(True)
    legend(fontsize=12, framealpha=1.0)

    savefig("ROC_curves_%.1f.pdf" % R_thr, bbox_inches="tight")

for R_thr in results[ensemble_size]["reldiag"].keys():
    fig = figure(figsize=(5, 3.5))
    ax = fig.gca()
    iax = inset_axes(ax, width="35%", height="20%", loc=4, borderpad=3.5)

    sample_size_min = []
    sample_size_max = []

    for i,lt in enumerate(leadtimes):
        reldiag = results[ensemble_size]["reldiag"][R_thr][lt]

        f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
        r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]

        mask = np.logical_and(np.isfinite(r), np.isfinite(f))

        ax.plot(r[mask], f[mask], color=linecolors[i], linestyle='-', marker='D', 
                label="%d minutes" % ((lt+1)*5))

        bd = 0.5 * (reldiag["bin_edges"][1] - reldiag["bin_edges"][0])
        iax.plot(reldiag["bin_edges"][:-1] + bd, reldiag["sample_size"], 
                 color=linecolors[i], linestyle='-', marker='D', ms=3)

        sample_size_min.append(int(max(floor(log10(min(reldiag["sample_size"]))), 1)))
        sample_size_max.append(int(ceil(log10(max(reldiag["sample_size"])))))

    iax.set_yscale("log", basey=10)
    iax.set_xticks(reldiag["bin_edges"])
    iax.set_xticklabels(["%.1f" % max(v, 1e-6) for v in reldiag["bin_edges"]])
    yt_min = min(sample_size_min)
    yt_max = max(sample_size_max)
    t = [pow(10.0, k) for k in range(yt_min, yt_max)]
    iax.set_yticks([int(t_) for t_ in t])
    iax.set_xlim(0.0, 1.0)
    iax.set_ylim(0.5*t[0], 5*t[-1])
    iax.set_ylabel("$\log_{10}$(samples)")
    iax.yaxis.tick_right()
    iax.yaxis.set_label_position("right")
    iax.tick_params(axis="both", which="major", labelsize=6)
    iax.grid(axis='y')

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(fontsize=12, framealpha=1.0)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed relative frequency")

    savefig("reldiags_%.1f.pdf" % R_thr, bbox_inches="tight")

for R_thr in results[ensemble_size]["rankhist"].keys():
    figure(figsize=(5, 3.5))

    r_max = 0.0
    for i,lt in enumerate(leadtimes):
        rankhist = results[ensemble_size]["rankhist"][R_thr][lt]
        r = ensscores.rankhist_compute(rankhist)
        r_max = max(r_max, np.max(r))
        x = np.linspace(0, 1, rankhist["num_ens_members"] + 1)
        x += 0.5 * (x[1] - x[0])
        plot(x, r, color=linecolors[i], linestyle='-', marker='D', 
             label="%d minutes" % ((lt+1)*5))

    xticks(x[::3] + (x[1] - x[0]), np.arange(1, len(x))[::3])
    xlim(0, 1+1.0/len(x))
    ylim(0, r_max*1.25)
    xlabel("Rank of observation (among ensemble members)")
    ylabel("Relative frequency")
    grid(True, axis='y')
    legend(fontsize=12, framealpha=1.0)

    savefig("rankhists_%.1f.pdf" % R_thr, bbox_inches="tight")
