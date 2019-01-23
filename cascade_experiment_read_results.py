#!/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.verification import ensscores
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Data dir
experiment_name = "new_cascade_mask_bps"
base_dir = "/scratch/lforesti/pysteps-data/output/" + experiment_name
fig_dir_out = "/users/lforesti/pysteps-publication/figures"

# Experiment parameters
cascade_levels = [1,8]
mask_method = ['incremental', None]

thrs_rel = [0.1, 2, 5]
thrs_rk = [0.1]
v_leadtimes = [60,120]
v_accum = 5

# Plot parameters
cols = ["C3", "C1", "C0", "C2"]
ncols = 2
nrows = 2
    
for v_leadtime in v_leadtimes:
   
    if ncols == nrows:
        plt.figure(figsize=(8.5,8.5))
    else:
        plt.figure(figsize=(12.5,8.5))
    
    pl=0

    #%% Rank histograms
    for thr in thrs_rk:
        pl+=1
        plt.subplot(nrows,ncols,pl)

        rankhists={}
        maxr = 0

        w = 0
        # LOOP over experiments
        for p in mask_method:
            for c in cascade_levels:
                
                # Read-in rank histogram file
                rankhists[w] = {}
                fn = base_dir + '/n_cascade_levels-%i/mask_method-%s/rankhist_%03i_%03i_thr%.1f.csv' % (c,p,v_leadtime,v_accum,thr)
                
                with open(fn, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for rows in reader:
                        k = rows[0]
                        v = rows[1]
                        v = ''.join(s for s in v if s.isdigit() or s in ["."," ","-","+","e"])
                        v = np.array([float(s) for s in v.split()])
                        rankhists[w].update({k:v})
                print(fn, '// N samples =', int(np.nansum(rankhists[w]['n'])))   
                
                # Compute frequencies                
                r = ensscores.rankhist_compute(rankhists[w])
                x = np.linspace(0, 1, rankhists[w]["num_ens_members"] + 1)
                
                # Plot rank histogram
                if w==0:
                    plt.plot(x, np.ones_like(x)*1/x.size, ":k")
                if p is not None:
                    p_label = 'with mask'
                else:
                    p_label = 'without mask'
                plt.plot(x, r, 'o-', color=cols[w], label="%i-levels %s" % (c,p_label))
                maxr = np.max((maxr, np.max(r)))
                
                w+=1
        
        # Plot decorations
        plt.legend(loc='lower left',fontsize=9)
        ax = plt.gca()
        ax.set_xticks(x[::3] + (x[1] - x[0]))
        ax.set_xticklabels(np.arange(1, len(x))[::3], fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, maxr*1.01)
        ax.set_ylim(0, 0.10)

        ax.set_xlabel("Rank of observation (among ensemble members)", fontsize=10)
        ax.set_ylabel("Relative frequency", fontsize=10) 
        if thr == 0.1:
            ax.set_title('Rank histogram')
        else:
            ax.set_title(r'Rank histogram $\geq$ %.1f mm/h' % thr)

        ax.grid(True, axis='y', ls=':')
    
    ##% Reliability diagrams
    for thr in thrs_rel:
        pl+=1
        plt.subplot(nrows,ncols,pl)
        
        reldiags={}
        maxr = 0

        w = 0
        # LOOP over experiments
        for p in mask_method:
            for c in cascade_levels:
            
                # Read-in reliability file
                reldiags[w] = {}
                fn = base_dir + '/n_cascade_levels-%i/mask_method-%s/reldiag_%03i_%03i_thr%.1f.csv' % (c,p,v_leadtime,v_accum,thr)
                print(fn)
                with open(fn, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for rows in reader:
                        k = rows[0]
                        v = rows[1]
                        v = ''.join(s for s in v if s.isdigit() or s in ["."," ","-","+","e"])
                        v = np.array([float(s) for s in v.split()])
                        reldiags[w].update({k:v})
                
                reldiag = reldiags[w]
                f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
                r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]
                mask = np.logical_and(np.isfinite(r), np.isfinite(f))
                
                # Plot reliability diagram
                if w == 0:
                    ax = plt.gca()
                
                if p is not None:
                    p_label = 'with mask'
                else:
                    p_label = 'without mask'
                ax.plot(r[mask], f[mask], "o-", color=cols[w], label="%i-levels %s" % (c,p_label), zorder=2)
                
                if w > 0:
                    ax.plot([0, 1], [0, 1], "k--", zorder=1)
                
                # Plot sharpness diagram into an inset figure.
                if w==0:
                    iax = inset_axes(ax, width="34%", height="18%", loc="upper left", borderpad=0.5)

                bw = reldiag["bin_edges"][2] - reldiag["bin_edges"][1]
                plt.plot(reldiag["bin_edges"][:-1], reldiag["sample_size"], color=cols[w])
                
                w+=1

        # Reliability plot decoration
        ax.legend(loc='lower right',fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.grid(True, ls=':')

        ax.set_xlabel("Forecast probability", fontsize=10)
        ax.set_ylabel("Observed relative frequency", fontsize=10)
        ax.set_title(r'Reliability for rainfall $\geq$ %.1f mm/h' % thr)
        
        # Sharpness plot decoration
        iax.set_yscale("log", basey=10)
        iax.set_xticks(reldiag["bin_edges"])
        iax.set_xticklabels(["%.1f" % max(v, 1e-6) for v in reldiag["bin_edges"]],rotation=90)
        yt_min = int(max(np.floor(np.log10(min(reldiag["sample_size"][:-1]))), 1))
        yt_max = int(np.ceil(np.log10(max(reldiag["sample_size"][:-1]))))
        t = [pow(10.0, k) for k in range(yt_min, yt_max)]

        iax.set_yticks([int(t_) for t_ in t])
        iax.set_xlim(0.0, 1.0)
        iax.set_ylim(t[0], 5*t[-1])
        iax.set_ylabel(r"log$_{10}$(samples)", fontsize=7)
        iax.yaxis.tick_right()
        iax.yaxis.set_label_position("right")
        iax.tick_params(axis="both", which="major", labelsize=6)
        
    plt.tight_layout()
    
    # Save plot
    filename_fig = fig_dir_out + "/" + experiment_name + "_mch_results_%03i_%03i.pdf"  % (v_leadtime,v_accum)
    plt.savefig(filename_fig, bbox="tight")
    print(filename_fig, 'saved.')
