#!/bin/env python

"""
Cascade experiment analysis precipitation field plots:
Pysteps ensembles are generated with/without cascade decomposition and with/without precipitation mask.
The nowcast fields are then plotted and analyzed using power spectra.
The nowcast can be done in either Eulerian or Lagrangian frame. Nowcast accumulations can also be computed and analyzed.
"""
from cartopy import crs
from matplotlib import gridspec
from os.path import join

import datetime
import matplotlib.pylab as plt
import matplotlib
from string import ascii_lowercase

matplotlib.use('Agg')
import numpy as np
import pickle
import os
import sys
import time

import pysteps as stp
from pysteps.noise.fftgenerators import build_2D_tapering_function

def create_dir(directory_path):
    """
    Create a directory if does not exists.
    """

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path)
        except Exception:

            raise Exception("Error creating directory: " + directory_path)

# Precipitation events
data_source   = "mch_hdf5"
events = ["201701311000", "201607111300"]
events = ["201701311000"]
compute = True

# Whether to analyze the rainfall accumulations or the final rainrate fields
accumulation        = False
adv_method          = "semilagrangian"
  
## Methods
oflow_method        = "lucaskanade"     # lucaskanade, darts, None
nwc_method          = "steps"
noise_method        = "nonparametric"   # parametric, nonparametric, ssft
bandpass_filter     = "gaussian"
decomp_method       = "fft"

## Forecast parameters
n_cascade_levels_l  = [1,8]
mask_method_l       = ['incremental', None]     # sprog, obs or incremental

n_prvs_times        = 2                 # use at least 9 with DARTS
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
seed                = 24                # for reproducibility

# Set the BPS motion perturbation parameters that are adapted to the Swiss domain
# vp_par  = (2.56338484, 0.3330941, -2.99714349) # mch only
# vp_perp = (1.31204508, 0.3578426, -1.02499891)
vp_par  = (2.31970635, 0.33734287, -2.64972861) # fmi+mch
vp_perp = (1.90769947, 0.33446594, -2.06603662)

if motion_pert == "bps":
    print("Using Swiss parameters for motion perturbation.")
    vel_pert_kwargs = {"p_pert_par":vp_par, "p_pert_perp":vp_perp}
else:
    print("Using default parameters for motion perturbation.")
    vel_pert_kwargs = {} # Will use the default parameters

# Plot parameters          
out_dir_figs= "./figures/"
create_dir(out_dir_figs)
fig_fmt = 'png'
dpi = 300
cartopy_scale = "10m"
cols = ["C3", "C1", "C0", "C2"]

if data_source == "mch_hdf5":   
    wavelength_ticks = [1024,512,256,128,64,32,16,8,4,2]
if data_source == "mch_hdf5":
    wavelength_ticks = [512,256,128,64,32,16,8,4,2]
    
## LOOP over precipitation events
for startdate_str in events:
    data_dir = join("./data", startdate_str)
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
    if compute:
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
        else:
            R_obs_accum = np.mean(R_obs, axis=0)
            valid_time_fx_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
            valid_time_obs_txt = valid_time_fx_txt
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
            valid_time_obs_txt = '0 min'
            valid_time_fx_txt = '+' + str(int(n_lead_times*ds.timestep)) + ' min'
        accum_txt = 'rain rate'
    
    ## Threshold the accumulations
    R_obs_accum[R_obs_accum < r_threshold] = 0.0
    
    # Generate figure
    scale=0.9
    plt.figure(figsize=(15*scale,8.5*scale))
    n_rows = 2
    n_cols = 3
    x_abcd = -0.15
    y_abcd = 1
    title_ftsize=18
    
    # Plot observed rainfall field
    lw = 1.0

    gs = gridspec.GridSpec(n_rows, n_cols)
    left = 0.1
    right = 0.92
    top = 0.97
    bottom = 0.1
    hspace = 0.15
    wspace = 0.1
    gs.update(left=left, right=right,
              wspace=wspace, hspace=hspace,
              top=top, bottom=bottom)

    gs_cbar = gridspec.GridSpec(1,1)
    gs_cbar.update(left=right+0.015, right=0.95,
                   wspace=wspace, hspace=hspace,
                   top=top*0.99, bottom=bottom*1.04)


    cax = plt.subplot(gs_cbar[0, 0])

    ax = plt.subplot(gs[0, 0])
    ax = stp.plt.plot_precip_field(R_obs_accum, title='', map="cartopy",
                                   geodata=metadata_obs,
                                   cartopy_scale=cartopy_scale, lw=lw,
                                   cartopy_subplot=gs[0, 0],
                                   colorbar=False)

    plt.text(0.02,0.98, valid_time_obs_txt, transform=ax.transAxes,
             fontsize=16, verticalalignment='top')

    #plt.text(x_abcd, y_abcd, 'a)', transform=ax.transAxes, fontsize=12)
    ax.set_title('a) Observed ' + accum_txt, color='k', fontsize=title_ftsize)
    
    # Compute the power spectra from the dBR field
    R_obs_accum[np.isnan(R_obs_accum)] = 0.0
    R_obs_accum_dbr = to_dB(R_obs_accum, metadata_obs)[0]
    
    # Remove rain/no-rain transition
    R_obs_accum_shift = stp.utils.remove_rain_norain_discontinuity(R_obs_accum_dbr)
    
    # Apply window function to reduce edge effects when rain touches the domain borders
    window = build_2D_tapering_function(R_obs_accum_shift.shape, win_type='flat-hanning')
    R_obs_accum_shift *= window
    
    # Compute FFT spectrum of observed rainfall field
    R_obs_accum_spectrum, fft_freq = stp.utils.rapsd(R_obs_accum_shift,
                                                     np.fft,
                                                     return_freq=True, d=1.0)


    ax = plt.subplot(gs[1,0])
    lw = 1.0
    stp.plt.plot_rapsd(fft_freq, R_obs_accum_spectrum, x_units='km',
                       y_units='dBR', wavelength_ticks=wavelength_ticks,
                       color='k', lw=2.0, label='Observations', ax=ax)
    
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

            file_prefix = str(mask_method) + "_" + str(n_cascade_levels)

            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Analyzing mask', precip_mask, 'and n_cascade_levels', n_cascade_levels)
            tic = time.time()
            
            if n_cascade_levels == 1:
                bandpass_filter = "uniform"
            else:
                bandpass_filter = "gaussian"

            if compute:
                ## Compute nowcast
                R_fct = nwc(R, UV, n_lead_times, n_ens_members,
                                   n_cascade_levels, kmperpixel=metadata["xpixelsize"]/1000,
                                   timestep=ds.timestep,  R_thr=metadata["threshold"],
                                   extrap_method=adv_method, decomp_method=decomp_method,
                                   bandpass_filter_method=bandpass_filter, num_workers=4,
                                   noise_method=noise_method, noise_stddev_adj=adjust_noise,
                                   ar_order=ar_order, conditional=conditional,
                                   mask_method=mask_method, mask_kwargs={'mask_rim':10},
                                   probmatching_method=prob_matching,
                                   vel_pert_method=motion_pert, vel_pert_kwargs=vel_pert_kwargs, seed=seed)

                ## if necessary, transform back all data to rainrates
                R_fct, metadata_fct = to_dB(R_fct, metadata, inverse=True)

                create_dir(data_dir)

                file_name = "R_fct_" + file_prefix
                np.save(join(data_dir, file_name),R_fct)

                file_name = "metadata_fct_" + file_prefix + ".pk"
                pickle.dump(metadata_fct, open("file_name", "wb"))
            else:
                file_name = "R_fct_" + file_prefix + ".npy"
                R_fct = np.load(join(data_dir, file_name))

                file_name = "metadata_fct_" + file_prefix + ".pk"
                metadata_fct = pickle.load(open("file_name", "rb"))


            ## LOOP over ensemble members
            for member in range(0, n_ens_members):
                # Compute accumulations or get intensity at last lead time
                if accumulation:
                    R_fct_accum = np.mean(R_fct[member,:,:,:], axis=0)
                else:
                    R_fct_accum = R_fct[member,-1,:,:]                       

                R_fct_accum[np.isnan(R_fct_accum)] = 0.0
                
                ## Threshold the accumulations
                R_fct_accum[R_fct_accum < r_threshold] = 0.0
                
                # Convert to dBR
                R_fct_accum_dbr = to_dB(R_fct_accum, metadata_fct)[0]
                
                # Remove rain/no-rain transition
                R_fct_accum_shift = stp.utils.remove_rain_norain_discontinuity(R_fct_accum_dbr)
                
                # Apply window
                R_fct_accum_shift *= window
                
                # Compute Fourier spectrum for each member
                if member == 0:
                    R_fct_accum_spectrum = stp.utils.rapsd(R_fct_accum_shift, np.fft, d=1.0)
                else:
                    R_fct_accum_spectrum += stp.utils.rapsd(R_fct_accum_shift, np.fft, d=1.0)
            
            # Compute average Fourier spectrum
            R_fct_accum_spectrum /= n_ens_members 
            
            ## Plot Fourier spectra 
            lw = 1.0 #0.5
            ax = plt.subplot(gs[1, 0])
            # ax = plt.subplot(n_rows,n_cols,n_cols+1)
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
                ax.set_title('d) Fourier spectra', fontsize=title_ftsize)
                ax.set_ylim([-10,60])
                plt.setp(ax.get_xticklabels(), fontsize=12)
                plt.setp(ax.get_yticklabels(), fontsize=12)
                ax.xaxis.label.set_size(14)
                ax.yaxis.label.set_size(14)
                # ax.text(x_abcd, y_abcd, 'd)', transform=ax.transAxes, fontsize=12)
                
            if p == n_cols+1:
                p = n_cols+2
            
            # Plot forecast rainfall fields

            print('p = ', p)
            print("i=",(p-1)//n_cols)
            print("j=", (p-1) % n_cols)
            print(n_rows, n_cols)


            if precip_mask:
                precip_mask_label = '+ precipitation mask'
            else:
                precip_mask_label = '- precipitation mask'
            stp.plt.plot_precip_field(R_fct_accum,
                                      title=str(n_cascade_levels) + ' cascade levels \n' + precip_mask_label,
                                      map="cartopy", geodata=metadata,
                                      cartopy_scale=cartopy_scale, lw=lw,
                                      cartopy_subplot=gs[(p-1)//n_cols,(p-1)%n_cols],
                                      colorbar=True,cax=cax)
            
            plot_label = chr(ord('a') + p-1)

            ax = plt.gca()
            plt.text(0.02,0.98, valid_time_fx_txt, transform=ax.transAxes, fontsize=16, verticalalignment='top')
            #plt.text(x_abcd, y_abcd, ascii_lowercase[p-1] + ')', transform=ax.transAxes, fontsize=12)
            ax.set_title(ascii_lowercase[p-1] + ') '+
                         title_str, color=cols[w], fontsize=title_ftsize)
            
            p+=1
            w+=1
            
            toc = time.time()
            print('Elapsed time for one parameter setting:', toc-tic)

    cax.title.set_fontsize(15)
    # Save plot
    figname = out_dir_figs + data_source[0:3] + '_' + startdate_str + '_multipanel_' + adv_method + "_" + str(int(n_lead_times)*ds.timestep) + 'min_accumulation-' + str(accumulation) + '.' + fig_fmt
    plt.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(figname, 'saved.')
    
print('Finished!')