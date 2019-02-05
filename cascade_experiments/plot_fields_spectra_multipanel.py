#!/bin/env python

"""
Cascade experiment analysis precipitation field plots:
Pysteps ensembles are generated with/without cascade decomposition and with/without precipitation mask.
The nowcast fields are then plotted and analyzed using power spectra.
The nowcast can be done in either Eulerian or Lagrangian frame. Nowcast accumulations can also be computed and analyzed.
"""

import datetime
import matplotlib.pylab as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import pickle
import os
import sys
import time
import pysteps as stp

# Precipitation events
data_source   = "mch_hdf5"
events = ["201701311000", "201607111300"]

# Whether to analyze the rainfall accumulations or the final rainrate fields
accumulation        = False
adv_method          = "eulerian"  # semilagrangian, eulerian

# Plot parameters          
out_dir_figs= "figures/"
fig_fmt = 'png'
dpi = 300
cartopy_scale = "10m"
cols = ["C3", "C1", "C0", "C2"]

## Methods
oflow_method        = "lucaskanade"     # lucaskanade, darts, None
nwc_method          = "steps"
noise_method        = "nonparametric"   # parametric, nonparametric, ssft
bandpass_filter     = "gaussian"
decomp_method       = "fft"

## Forecast parameters
n_cascade_levels_l  = [1,8]
mask_method_l       = ['incremental', None]     # sprog, obs or incremental

n_prvs_times        = 3                 # use at least 9 with DARTS
n_lead_times        = 12
n_ens_members       = 24
ar_order            = 2
r_threshold         = 0.1               # rain/no-rain threshold [mm/h]
zero_value_dbr      = -15
adjust_noise        = None              # Whether to adjust the noise fo the cascade levels
prob_matching       = "cdf"

conditional         = False
motion_pert         = 'bps'
unit                = "mm/h"            # mm/h or dBZ
transformation      = "dB"              # None or dB
seed                = 42                # for reproducibility

# Set the BPS motion perturbation parameters that are adapted to the Swiss domain
if motion_pert == "bps":
    print("Using Swiss parameters for motion perturbation.")
    vel_pert_kwargs = {"p_pert_par":(2.56,0.33,-3.0), "p_pert_perp":(1.31,0.36,-1.02)}
else:
    print("Using default parameters for motion perturbation.")
    vel_pert_kwargs = {} # Will use the default parameters

## LOOP over precipitation events
for startdate_str in events:
    
    # Read-in the data
    print('Read the data...')
    startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

    ## import data specifications
    ds = stp.rcparams.data_sources[data_source]

    ## find radar field filenames
    input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                      ds.fn_ext, ds.timestep, n_prvs_times, 0)

    ## read radar field files
    importer = stp.io.get_method(ds.importer, method_type="importer")
    R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
    Rmask = np.isnan(R)

    # Prepare input files
    print("Prepare the data...")

    ## if necessary, convert to rain rates [mm/h]
    converter = stp.utils.get_method("mm/h")
    R, metadata = converter(R, metadata)

    ## threshold the data
    R[R<r_threshold] = 0.0
    metadata["threshold"] = r_threshold
    
    ## convert the data
    converter = stp.utils.get_method(unit)
    R, metadata = converter(R, metadata)
    
    ## transform the data
    to_dB = stp.utils.get_method(transformation)
    R, metadata = to_dB(R, metadata, zerovalue=zero_value_dbr)

    ## set NaN equal to zero
    R[~np.isfinite(R)] = metadata["zerovalue"]

    # Compute motion field
    of = stp.motion.get_method(oflow_method)
    UV = of(R)

    # Get the nowcast method
    nwc = stp.nowcasts.get_method(nwc_method)

    ## find the verifying observations
    input_files_verif = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                            ds.fn_ext, ds.timestep, 0, n_lead_times)

    ## read observations
    R_obs, _, metadata_obs = stp.io.read_timeseries(input_files_verif, importer,
                                                    **ds.importer_kwargs)
    R_obs = R_obs[1:,:,:]
    metadata_obs["timestamps"] = metadata_obs["timestamps"][1:]

    ## if necessary, convert to rain rates [mm/h]
    R_obs, metadata_obs = converter(R_obs, metadata_obs)

    ## Threshold the rainrates
    R_obs[R_obs < r_threshold] = 0.0
    metadata_obs["threshold"] = r_threshold
    
    # Compute observed accumulations
    if accumulation:
        if adv_method == 'eulerian':
            print('Setting not available.')
            sys.exit(1)
            # R_mmhr = to_dB(R[-1,:,:], metadata, inverse=True)[0]
            # R_obs_accum = np.mean(R_mmhr, axis=0)
            # accum_txt = '-60-0' + str(int(n_lead_times/12)) + ' min accumulation'
        else:
            R_obs_accum = np.mean(R_obs, axis=0)
            valid_time_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
        accum_txt = 'accumulation'
        
    else:
        # Get observed intensities
        if adv_method == 'eulerian':
            R_obs_accum_dbr = R[-1,:,:]
            R_obs_accum = to_dB(R_obs_accum_dbr, metadata, inverse=True)[0]
            valid_time_obs_txt = '0 min'
            valid_time_fx_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
        else:
            R_obs_accum = R_obs[-1,:,:]
            valid_time_obs_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
            valid_time_fx_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
        accum_txt = 'rain rate'
    
    ## Threshold the accumulations
    R_obs_accum[R_obs_accum < r_threshold] = 0.0
    
    # Generate figure
    plt.figure(figsize=(17,8.5))
    n_rows = 2
    n_cols = 3
    x_abcd = -0.15
    y_abcd = -0.12
    title_ftsize=18
    
    # Plot observed rainfall field
    lw = 1.0
    plt.subplot(n_rows,n_cols,1)
    ax = stp.plt.plot_precip_field(R_obs_accum, title='', map="cartopy", geodata=metadata_obs, cartopy_scale=cartopy_scale, lw=lw, cartopy_subplot=(n_rows,n_cols,1), colorbar=True)
    plt.text(0.02,0.98, valid_time_obs_txt, transform=ax.transAxes, fontsize=16, verticalalignment='top')
    plt.text(x_abcd, y_abcd, 'a)', transform=ax.transAxes, fontsize=12)
    ax.set_title('Observed ' + accum_txt, color='k', fontsize=title_ftsize)
    
    # Compute the power spectra from the dBR field
    R_obs_accum[np.isnan(R_obs_accum)] = 0.0
    R_obs_accum = to_dB(R_obs_accum, metadata_obs)[0]
    
    # Remove rain/no-rain transition
    R_obs_accum_shift = stp.utils.remove_rain_norain_discontinuity(R_obs_accum)
    
    # Compute FFT spectrum of observed rainfall field
    R_obs_accum_spectrum, fft_freq = stp.utils.rapsd(R_obs_accum_shift, np.fft, return_freq=True, d=1.0)
    
    ax = plt.subplot(n_rows, n_cols, n_cols+1)
    lw = 1.0
    wavelength_ticks = [512,256,128,64,32,16,8,4,2]
    stp.plt.plot_rapsd(fft_freq, R_obs_accum_spectrum, x_units='km', y_units='dBR', wavelength_ticks=wavelength_ticks, color='k', lw=lw, label='Observations', ax=ax)
    
    p=2 # plot index
    w=0 # color index
    ## LOOP over mask
    for mask_method in mask_method_l:
        if mask_method is None:
            precip_mask = False
        else:
            precip_mask = True
            
        ## LOOP over cascade levels
        for n_cascade_levels in n_cascade_levels_l:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Analyzing mask', precip_mask, 'and n_cascade_levels', n_cascade_levels)
            tic = time.time()
            
            if n_cascade_levels == 1:
                bandpass_filter = "uniform"
            else:
                bandpass_filter = "gaussian"
            
            ## Compute nowcast    
            R_fct = nwc(R, UV, n_lead_times, n_ens_members,
                               n_cascade_levels, kmperpixel=metadata["xpixelsize"]/1000,
                               timestep=ds.timestep,  R_thr=metadata["threshold"],
                               extrap_method=adv_method, decomp_method=decomp_method,
                               bandpass_filter_method=bandpass_filter,
                               noise_method=noise_method, noise_stddev_adj=adjust_noise,
                               ar_order=ar_order, conditional=conditional,
                               mask_method=mask_method,
                               probmatching_method=prob_matching, 
                               vel_pert_method=motion_pert, vel_pert_kwargs=vel_pert_kwargs, seed=seed)

            ## if necessary, transform back all data to rainrates
            R_fct, metadata_fct = to_dB(R_fct, metadata, inverse=True)
            
            ## LOOP over ensemble members
            for member in range(0, n_ens_members):
                # Compute accumulations or get intensity at last lead time
                if accumulation:
                    R_fct_accum = np.mean(R_fct[member,:,:,:], axis=0)
                else:
                    R_fct_accum = R_fct[member,-1,:,:]                       

                R_fct_accum[np.isnan(R_fct_accum)] = 0.0
                
                ## threshold the accumulations
                R_fct_accum[R_fct_accum < r_threshold] = 0.0
                
                # Compute Fourier spectrum for each member
                R_fct_accum = to_dB(R_fct_accum, metadata_fct)[0]
                
                # Remove rain/no-rain transition
                R_fct_accum_shift = stp.utils.remove_rain_norain_discontinuity(R_fct_accum)
    
                if member == 0:
                    R_fct_accum_spectrum = stp.utils.rapsd(R_fct_accum_shift, np.fft, d=1.0)
                else:
                    R_fct_accum_spectrum += stp.utils.rapsd(R_fct_accum_shift, np.fft, d=1.0)
            
            # Compute average Fourier spectrum
            R_fct_accum_spectrum /= n_ens_members 
            
            ## Plot Fourier spectra 
            lw = 1.0 #0.5
            ax = plt.subplot(n_rows,n_cols,n_cols+1)
            if (n_cascade_levels == 1) and  (precip_mask == True):
                title_str = '1 levels + mask'
                stp.plt.plot_rapsd(fft_freq, R_fct_accum_spectrum, color=cols[w], lw=1.0, label=title_str, ax=ax)
            if (n_cascade_levels > 1) and  (precip_mask == True):
                title_str = str(n_cascade_levels) + ' levels + mask'
                stp.plt.plot_rapsd(fft_freq, R_fct_accum_spectrum, color=cols[w], lw=1.0, label=title_str, ax=ax)
            if (n_cascade_levels == 1) and (precip_mask == False):
                title_str = '1 levels - mask'
                stp.plt.plot_rapsd(fft_freq, R_fct_accum_spectrum, color=cols[w], lw=1.0, label=title_str, ax=ax)
            if (n_cascade_levels > 1) and (precip_mask == False): 
                title_str = str(n_cascade_levels) + ' levels - mask'
                stp.plt.plot_rapsd(fft_freq, R_fct_accum_spectrum, color=cols[w], lw=1.0, label=title_str, ax=ax)
            if p == len(n_cascade_levels_l)*len(mask_method_l) + 2:
                # Create legend
                leg = ax.legend(fontsize=11, loc='lower left')
                c=-1
                for text in leg.get_texts():
                    if c >= 0:
                        plt.setp(text, color=cols[c])
                    c+=1
                # Decorate plot
                ax.set_title('Fourier spectra', fontsize=title_ftsize)
                ax.set_ylim([-20,60])
                plt.setp(ax.get_xticklabels(), fontsize=12)
                plt.setp(ax.get_yticklabels(), fontsize=12)
                ax.xaxis.label.set_size(14)
                ax.yaxis.label.set_size(14)
                ax.text(x_abcd, y_abcd, 'd)', transform=ax.transAxes, fontsize=12)
                
            if p == n_cols+1:
                p = n_cols+2
            
            # Plot forecast rainfall fields
            plt.subplot(n_rows,n_cols,p)
            print('p = ', p)
            if precip_mask:
                precip_mask_label = '+ precipitation mask'
            else:
                precip_mask_label = '- precipitation mask'
            stp.plt.plot_precip_field(R_fct_accum, title=str(n_cascade_levels) + ' cascade levels \n' + precip_mask_label, map="cartopy", geodata=metadata, cartopy_scale=cartopy_scale, lw=lw, cartopy_subplot=(n_rows,n_cols,p), colorbar=True)
            
            plot_label = chr(ord('a') + p-1)
            ax = plt.gca()
            plt.text(0.02,0.98, valid_time_fx_txt, transform=ax.transAxes, fontsize=16, verticalalignment='top')
            plt.text(x_abcd, y_abcd, plot_label + ')', transform=ax.transAxes, fontsize=12)
            ax.set_title(title_str, color=cols[w], fontsize=title_ftsize)
            
            p+=1
            w+=1
            
            toc = time.time()
            print('Elapsed time for one parameter setting:', toc-tic)
    
    # Save plot
    if accumulation:
        figname = out_dir_figs + startdate_str + '_multipanel_' + adv_method + "_" + str(int(n_lead_times)*ds.timestep) + 'min_accumulation.' + fig_fmt
    else:
        figname = out_dir_figs + startdate_str + '_multipanel_' + adv_method + "_" + str(int(n_lead_times)*ds.timestep) + 'min_intensity.' + fig_fmt
    plt.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(figname, 'saved.')
    
print('Finished!')