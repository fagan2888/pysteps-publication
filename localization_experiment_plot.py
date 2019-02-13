"""
Section 6.3
This experiment investigates the impact of localization on the quality  of  the  
nowcast.  For  localization  we  intend  the  use of  a  subset  of  the  
observations  in  order  to  estimate  model parameters that are distributed in 
space. The short-space approach used in Nerini et al. (2017) is generalized to 
the whole nowcasting system. This essenially boils down to a moving window 
localization of the nowcasting procedure, whereby all parameters are estimated 
over a subdomain of prescribed size.

This script plots the results.
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *
import matplotlib.gridspec as gridspec
import pickle
from pysteps.verification import ensscores, probscores

linecolors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
linestyles = ['-', '-', '-', '-', '-']
markers = ['o', 'o', 'o', 'o', 'o']
fixedleadtimes = [5, 11, 17] # timesteps
minmaxleadtimes = [10, 90] # min
R_thrs = [0.1, 1.0, 5.0, 10.0]

with open("window_size_results.dat", "rb") as f:
    results = pickle.load(f)
    
for R_thr in R_thrs:
        
    # ROC curves
    for lt in fixedleadtimes:

        figure()

        plot([0, 1], [0, 1], "k--")

        for i,ws in enumerate(results.keys()):
            ROC = results[ws]["ROC"][R_thr][lt]
            POFD,POD,area = probscores.ROC_curve_compute(ROC, compute_area=True)
            plot(POFD, POD, color=linecolors[i], linestyle='-', marker='D', 
                 label="%d km (area=%.3f)" % (ws, area))

        xlim(0, 1)
        ylim(0, 1)
        xlabel("False alarm rate (POFD)")
        ylabel("Probability of detection (POD)")
        grid(True, ls=':')
        legend(fontsize=12)

        savefig("figures/window_size_ROC_curves_%dmin_%03.1fmm.pdf" % ((lt+1)*5, R_thr), bbox_inches="tight")
        
    # Rank hist

    for lt in fixedleadtimes:
        figure(figsize=(6,11))
        gs = gridspec.GridSpec(2, 1)
        ax = subplot(gs[0])

        r_max = 0.0
        for i,ws in enumerate(results.keys()):
            rankhist = results[ws]["rankhist"][R_thr][lt]
            r = ensscores.rankhist_compute(rankhist)
            r_max = max(r_max, np.max(r))
            x = np.linspace(0, 1, rankhist["num_ens_members"] + 1)
            x += 0.5 * (x[1] - x[0])
            ax.plot(x, r, color=linecolors[i], linestyle='-', marker='D', 
                 label="%d km" % ws)
                 
        ax.plot(x, np.ones_like(x)*1/x.size, "k--")

        xticks(x[::3] + (x[1] - x[0]), np.arange(1, len(x))[::3])
        ax.set_xlim(0, 1+1.0/len(x))
        ax.set_ylim(0, r_max*1.25)
        ax.set_xlabel("Rank of observation (among ensemble members)")
        ax.set_ylabel("Relative frequency")
        ax.set_title("(a)")
        ax.grid(True, axis='y', ls=':')
        legend(fontsize=12)

        # savefig("figures/window_size_rankhists_%dmin_%03.1fmm.pdf" % ((lt+1)*5, R_thr), bbox_inches="tight")
        
    # Reliability diagrams

    # for lt in fixedleadtimes:
        # fig = figure()
        # ax = fig.gca()
        ax = subplot(gs[1])
        iax = inset_axes(ax, width="35%", height="20%", loc=4, borderpad=3.5)

        sample_size_min = []
        sample_size_max = []

        for i,ws in enumerate(results.keys()):
            reldiag = results[ws]["reldiag"][R_thr][lt]

            f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
            r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]

            mask = np.logical_and(np.isfinite(r), np.isfinite(f))

            ax.plot(r[mask], f[mask], color=linecolors[i], linestyle='-', marker='D', 
                    label="%d km" % ws)

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
        iax.grid(axis='y', ls=':')

        ax.plot([0, 1], [0, 1], "k--")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, ls=':')
        ax.legend(fontsize=12)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed relative frequency")
        ax.set_title("(b)")

        savefig("figures/window_size_rankhist-reldiags_%dmin_%03.1fmm.pdf" % ((lt+1)*5, R_thr), bbox_inches="tight")
        
        
    # ROC areas

    ROC_areas = dict([(ws, []) for ws in results.keys()])
    OPs = dict([(ws, []) for ws in results.keys()])

    for ws in results.keys():
      if not ws in ROC_areas.keys():
          ROC_areas[ws] = []
      for lt in sorted(results[ws]["ROC"][R_thr].keys()):
          a = probscores.ROC_curve_compute(results[ws]["ROC"][R_thr][lt], compute_area=True)[2]
          ROC_areas[ws].append(a)
      for lt in sorted(results[ws]["rankhist"][R_thr].keys()):
          rh = ensscores.rankhist_compute(results[ws]["rankhist"][R_thr][lt])
          OP = (rh[0] + rh[-1]) / sum(rh)
          OPs[ws].append(OP) 

    fig = figure()
    ax = fig.gca()
    for i,ws in enumerate(ROC_areas.keys()):
        leadtimes = (arange(len(ROC_areas[ws])) + 1) * 5
        ax.plot(leadtimes, ROC_areas[ws], ls=linestyles[i], marker=markers[i], 
                color=linecolors[i], label="%d km" % ws, lw=2, ms=6)
    ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
    # ax.set_ylim(0.82, 0.98)
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_xlabel("Lead time (minutes)", fontsize=12)
    ax.set_ylabel("ROC area", fontsize=12)

    fig.savefig("figures/window_size_ROC_areas_%03.1fmm.pdf" % R_thr, bbox_inches="tight")

    # OP

    fig = figure()
    ax = fig.gca()
    for i,ws in enumerate(OPs.keys()):
        leadtimes = (arange(len(OPs[ws])) + 1) * 5
        ax.plot(leadtimes, OPs[ws], ls=linestyles[i], marker=markers[i], 
                color=linecolors[i], label="%d km" % ws, lw=2, ms=6)
    ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
    # ax.set_ylim(0.01, 0.675)
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_xlabel("Lead time (minutes)", fontsize=12)
    ax.set_ylabel("Percentage of outliers", fontsize=12)

    fig.savefig("figures/window_size_OPs_%03.1fmm.pdf" % R_thr, bbox_inches="tight")
