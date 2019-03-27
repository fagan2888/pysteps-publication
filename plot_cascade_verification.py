#!/bin/env python

"""
Cascade verification plots:
The script reads the file generated by run_cascade_verification.py and plots
verification results for different lead times and rainfall thresholds.
"""

import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *
import pickle
from pysteps.verification import ensscores, probscores

# Parameters
filename_verif = "data/mch_cascade_results_accum05_crps_spread.dat"

v_leadtimes = [5, 15, 30, 60]
minmaxleadtimes = [5, 60] # min

R_thrs = [0.1, 1.0, 5.0]
v_scales_km = [1, 10, 40]

spread_skill_metric = "RMSE_add"

# Plot parameters
basename_figs = "cascade"
fmt = "pdf"
ftsize_title = 14
linecolors = ["C3", "C0", "C1", "C2", "C4", "C5", "C6"]
linestyles = ['-', '-', '-', '-', '-']
markers = ['o', 'o', 'o', 'o', 'o']

with open(filename_verif, "rb") as f:
    results = pickle.load(f)
    print(f, 'read.')
    
    # Get metadata on accumulation
    if "metadata" in results:
        v_accu = results["metadata"]["v_accu_min"]
        v_leadtimes_avail = results["metadata"]["v_leadtimes"]
        del results["metadata"]
    else:
        v_accu = 5
    
    # Get keys of experiments (without metadata)
    results_keys = results.keys()
    print("Available experiments:\n", list(results_keys))
    
    # Only plot available lead times
    v_leadtimes = list(set(v_leadtimes_avail) & set(v_leadtimes))
    
    skill_varname = spread_skill_metric + "_skill"
    spread_varname = spread_skill_metric + "_spread"
    
for R_thr in R_thrs:
    print("Rainfall threshold:", R_thr)
    for scale_km in v_scales_km:    
        # ROC curves
        for lt in v_leadtimes:
            figure(figsize=(5, 3.5))
            plot([0, 1], [0, 1], "k--")
            
            i=0
            for i,exp in enumerate(results_keys):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                ROC = results[c,m]["ROC"][R_thr][scale_km][lt]
                POFD,POD,area = probscores.ROC_curve_compute(ROC, compute_area=True)
                plot(POFD, POD, color=linecolors[i], linestyle='-', marker='D', 
                     label="%i levels %s mask (area=%.3f)" % (c, m_str, area))

            xlim(0, 1)
            ylim(0, 1)
            xlabel("False alarm rate (POFD)")
            ylabel("Probability of detection (POD)")
            title("ROC curves +%i min R > %.1f mm/h" % (lt, R_thr), fontsize=ftsize_title)
            title("ROC curves for R > %.1f mm/h" % (R_thr), fontsize=ftsize_title)
            grid(True, ls=':')
            legend(fontsize=9)
            
            figname = "figures/%s_ROC_curves_accu%02imin_scale%03ikm_%02imin_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, lt, R_thr, fmt)
            savefig(figname, bbox_inches="tight")
            print(figname, "saved.")
            
        # Reliability diagrams
        for lt in v_leadtimes:
            fig = figure(figsize=(5, 3.5))
            ax = fig.gca()
            #iax = inset_axes(ax, width="38%", height="30%", loc=4, borderpad=3.5) # bbox_to_anchor=(0.55, 0.05, 1, 1), bbox_transform=ax.transAxes
            iax = ax.inset_axes([0.53, 0.12, 0.33, 0.33], transform=ax.transAxes)
            
            sample_size_min = []
            sample_size_max = []

            for i,exp in enumerate(results_keys):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                reldiag = results[c,m]["reldiag"][R_thr][scale_km][lt]

                f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
                r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]

                mask = np.logical_and(np.isfinite(r), np.isfinite(f))
                
                # Plot reliability
                ax.plot(r[mask], f[mask], color=linecolors[i], linestyle='-', marker='D', 
                        label="%i levels %s mask" % (c, m_str))

                # Plot sharpness diagram
                bd = 0.5 * (reldiag["bin_edges"][1] - reldiag["bin_edges"][0])
                iax.plot(reldiag["bin_edges"][:-1] + bd, reldiag["sample_size"], 
                         color=linecolors[i], linestyle='-', ms=3)

                sample_size_min.append(int(max(floor(log10(min(reldiag["sample_size"]))), 1)))
                sample_size_max.append(int(ceil(log10(max(reldiag["sample_size"])))))

            iax.set_yscale("log", basey=10)
            iax.set_xticks(reldiag["bin_edges"])
            iax.set_xticklabels(["%.1f" % max(v, 1e-6) for v in reldiag["bin_edges"]], rotation=90)
            yt_min = min(sample_size_min)
            yt_max = max(sample_size_max)
            t = [pow(10.0, k) for k in range(yt_min, yt_max)]
            iax.set_yticks([int(t_) for t_ in t])
            iax.set_xlim(0.0, 1.0)
            iax.set_ylim(0.5*t[0], 5*t[-1])
            iax.set_ylabel("$\log_{10}$(samples)")
            iax.yaxis.tick_right()
            iax.yaxis.set_label_position("right")
            iax.tick_params(axis="both", which="major", labelsize=8)
            iax.grid(axis='y', ls=':')

            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, ls=':')
            ax.legend(fontsize=9, loc="upper left")
            ax.set_xlabel("Forecast probability")
            ax.set_ylabel("Observed relative frequency")
            title_str = "Reliability for R > %.1f mm/h" % (R_thr)
            if R_thr == 0.1 and scale_km == 1 and lt == 60:
                title_str = "(c) " + title_str
            if R_thr == 1.0 and scale_km == 1 and lt == 60:
                title_str = "(d) " + title_str
            ax.set_title(title_str, fontsize=ftsize_title)
            
            figname = "figures/%s_reldiags_accu%02imin_scale%03ikm_%02imin_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, lt, R_thr, fmt)
            savefig(figname, bbox_inches="tight")
            print(figname, "saved.")
            
        # Rank hist
        for lt in v_leadtimes:
            figure(figsize=(5, 3.5))

            r_max = 0.0
            for i,exp in enumerate(results_keys):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                rankhist = results[c,m]["rankhist"][R_thr][scale_km][lt]
                r = ensscores.rankhist_compute(rankhist)
                r_max = max(r_max, np.max(r))
                x = np.linspace(0, 1, rankhist["num_ens_members"] + 1)
                x += 0.5 * (x[1] - x[0])
                plot(x, r, color=linecolors[i], linestyle='-', 
                     label="%i levels %s mask" % (c, m_str))
                     
            plot(x, np.ones_like(x)*1/x.size, "k--", lw=0.8)

            xticks(x[::3] + (x[1] - x[0]), np.arange(1, len(x))[::3])
            xlim(0, 1+1.0/len(x))
            ylim(0, r_max*1.25) 
            
            xlabel("Rank of observation (among ensemble members)")
            ylabel("Relative frequency")
            title_str = "Rank histograms"
            if R_thr == 0.1 and scale_km == 1 and lt == 60:
                title_str = "(a) " + title_str
            title(title_str, fontsize=ftsize_title)
            grid(True, axis='y', ls=':')
            legend(fontsize=9)
            
            figname = "figures/%s_rankhists_accu%02imin_scale%03ikm_%02imin_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, lt, R_thr, fmt)
            savefig(figname, bbox_inches="tight")
            print(figname, "saved.")
        
        ## Plot scores vs lead times
        print("++++ score vs time plots ++++")
        if len(v_leadtimes) > 1:
            
            # Get and compute scores
            ROC_areas = dict([(exp, []) for exp in results_keys])
            OPs = dict([(exp, []) for exp in results_keys])
            for i,exp in enumerate(results_keys):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                if not exp in ROC_areas.keys():
                    ROC_areas[exp] = []
                for lt in sorted(results[c,m]["ROC"][R_thr][scale_km].keys()):
                    a = probscores.ROC_curve_compute(results[c,m]["ROC"][R_thr][scale_km][lt], compute_area=True)[2]
                    ROC_areas[exp].append(a)
                for lt in sorted(results[c,m]["rankhist"][R_thr][scale_km].keys()):
                    rh = ensscores.rankhist_compute(results[c,m]["rankhist"][R_thr][scale_km][lt])
                    OP = (rh[0] + rh[-1]) / sum(rh)
                    OPs[c,m].append(OP) 
            
            ## Plots
            # ROC areas
            fig = figure(figsize=(5, 3.5))
            ax = fig.gca()
            for i,exp in enumerate(ROC_areas.keys()):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                leadtimes = (arange(len(ROC_areas[c,m])) + 1) * 5
                ax.plot(v_leadtimes_avail, ROC_areas[exp], ls=linestyles[i], marker=markers[i], 
                        color=linecolors[i], label="%i levels %s mask" % (c, m_str), lw=2, ms=6)
            ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
            # ax.set_ylim(0.82, 0.98)
            ax.grid(True)
            ax.legend(fontsize=9)
            ax.set_xlabel("Lead time [minutes]")
            ax.set_ylabel("ROC area")
            title("ROC area for R > %.1f mm/h" % (R_thr), fontsize=ftsize_title)
            
            figname = "figures/%s_ROC_areas_accu%02imin_scale%03ikm_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, R_thr, fmt)
            fig.savefig(figname, bbox_inches="tight")
            print(figname, 'saved.')

            # Outlier proportion (OP)
            fig = figure(figsize=(5, 3.5))
            ax = fig.gca()
            for i,exp in enumerate(results_keys):
                c = exp[0]
                m = exp[1]
                m_str = "-" if m == None else "+"
                leadtimes = (arange(len(OPs[c,m])) + 1) * 5
                ax.plot(v_leadtimes_avail, OPs[c,m], ls=linestyles[i], marker=markers[i], 
                        color=linecolors[i], label="%i levels %s mask" % (c, m_str), lw=2, ms=6)
            ax.plot(v_leadtimes_avail, np.ones(len(leadtimes))*(2.0/(rankhist["num_ens_members"] + 1)), 'k--')
            ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
            # ax.set_ylim(0.01, 0.675)
            ax.grid(True)
            ax.legend(fontsize=9)
            ax.set_xlabel("Lead time [minutes]")
            ax.set_ylabel("Percentage of outliers")
            title("Outlier proportion for R > %.1f mm/h" % (R_thr), fontsize=ftsize_title)
            
            figname = "figures/%s_OPs_accu%02imin_scale%03ikm_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, R_thr, fmt)
            fig.savefig(figname, bbox_inches="tight")
            print(figname, 'saved.')
            
            # Only for lowest threshold
            if R_thr == R_thrs[0]:
                CRPSs = dict([(exp, []) for exp in results_keys])
                skills = dict([(exp, []) for exp in results_keys])
                spreads = dict([(exp, []) for exp in results_keys])
                
                # Get score values
                for i,exp in enumerate(results_keys):
                    c = exp[0]
                    m = exp[1]
                    m_str = "-" if m == None else "+"
                    for lt in sorted(results[c,m]["CRPS"][R_thr][scale_km].keys()):
                        a = probscores.CRPS_compute(results[c,m]["CRPS"][R_thr][scale_km][lt])
                        CRPSs[exp].append(a)
                    for lt in sorted(results[c,m][skill_varname][R_thr][scale_km].keys()):
                        a = results[c,m][skill_varname][R_thr][scale_km][lt]["sum"]/results[c,m][skill_varname][R_thr][scale_km][lt]["n"]
                        skills[exp].append(a)
                    for lt in sorted(results[c,m][spread_varname][R_thr][scale_km].keys()):
                        a = results[c,m][spread_varname][R_thr][scale_km][lt]["sum"]/results[c,m][spread_varname][R_thr][scale_km][lt]["n"]
                        spreads[exp].append(a)
                
                # CRPS
                fig = figure(figsize=(5, 3.5))
                ax = fig.gca()
                for i,exp in enumerate(results_keys):
                    c = exp[0]
                    m = exp[1]
                    m_str = "-" if m == None else "+"
                    ax.plot(v_leadtimes_avail, CRPSs[c,m], ls=linestyles[i], marker=markers[i], 
                            color=linecolors[i], label="%i levels %s mask" % (c, m_str), lw=2, ms=6)
                ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
                # ax.set_ylim(0.01, 0.675)
                ax.grid(True)
                ax.legend(fontsize=9)
                ax.set_xlabel("Lead time [minutes]")
                ax.set_ylabel("CRPS")
                title("CRPS", fontsize=ftsize_title)
                
                figname = "figures/%s_CRPSs_accu%02imin_scale%03ikm_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, R_thr, fmt)
                fig.savefig(figname, bbox_inches="tight")
                print(figname, 'saved.')
                
                # Spread-skill
                fig = figure(figsize=(5, 3.5))
                ax = fig.gca()
                for i,exp in enumerate(results_keys):
                    c = exp[0]
                    m = exp[1]
                    m_str = "-" if m == None else "+"
                    ax.plot(v_leadtimes_avail, skills[c,m], ls="-", 
                            color=linecolors[i], label="%i levels %s mask" % (c, m_str), lw=2, ms=6)
                    ax.plot(v_leadtimes_avail, spreads[c,m], ls=":", 
                            color=linecolors[i], label="%i levels %s mask" % (c, m_str), lw=2, ms=6)
                ax.set_xlim(minmaxleadtimes[0], minmaxleadtimes[1])
                # ax.set_ylim(0.01, 0.675)
                ax.grid(True)
                lines = ax.get_lines()
                legend1 = plt.legend([lines[i] for i in [0,2,4,6]], [lines[i].get_label() for i in [0,2,4,6]], loc="lower right", fontsize=9)
                legend2 = plt.legend([lines[i] for i in [0,1]], ["RMSE", "Spread"], loc="upper left", fontsize=9)
                ax.add_artist(legend1)
                ax.add_artist(legend2)
                ax.set_xlabel("Lead time [minutes]")
                ax.set_ylabel(r"RMSE and spread [mm h$^{-1}$]")
                title_str = "Spread-error relationship"
                if R_thr == 0.1 and scale_km == 1:
                    title_str = "(b) " + title_str
                title(title_str, fontsize=ftsize_title)
                
                figname = "figures/%s_spreads-skills_accu%02imin_scale%03ikm_%03.1fmm.%s" % (basename_figs, v_accu, scale_km, R_thr, fmt)
                fig.savefig(figname, bbox_inches="tight")
                print(figname, 'saved.')
        print('---------------')
print("Finished")