#!/bin/env python

"""
Script to verify spatial structure of SPROG filtering, ensemble members and ensemble mean nowcasts.
"""

from datetime import datetime, timedelta
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import numpy as np
import sys

import pysteps as stp
from pysteps import rcparams
from pysteps.noise.fftgenerators import build_2D_tapering_function
nc = stp.nowcasts.steps.forecast

# Import or define precip events
sys.path.insert(0,'..')
import precipevents

data_source = "mch_hdf5"
if data_source == "fmi":
    # all events
    precipevents = precipevents.fmi
    # only one event (comment out if you want all events)
    precipevents = [("201609281600", "201609281700")]
if data_source == "mch_hdf5":
    # all events
    precipevents = precipevents.mch
    # only 1 event (comment out if you want all events)
    precipevents = [("201701311000", "201701311000")]

# Experiment parameters (do not change order!) ['STOCH','MEAN','SPROG']
nowcast_types       = ['STOCH']

# Forecast parameters
timestep_run        = 240
num_prev_files      = 5
n_lead_times        = 6
n_members           = 2

n_levels            = 8
ar_order            = 2
filter              = "nonparametric"
mask_method         = "incremental"
pmatching_method    = "cdf"
zero_value_dbr      = -15
noise_stddev_adj    = None
conditional_stats   = False
seed                = 24

of_method           = "lucaskanade"
extrap_method       = "semilagrangian"

bandpass_filter = 'uniform' if (n_levels == 1) else 'gaussian'

# Figure parameters
animate             = True
nloops              = 2
plot_leadtimes      = [0,2,5,11,23]

plot_leadtimes = np.array(plot_leadtimes)
plot_leadtimes = plot_leadtimes[plot_leadtimes < n_lead_times]
if data_source == "fmi":
    wavelength_ticks = [1024,512,256,128,64,32,16,8,4,2]
if data_source[0:3] == "mch":
    wavelength_ticks = [512,256,128,64,32,16,8,4,2]
fmt = 'pdf'

# Loop over events
datasource = rcparams.data_sources[data_source]
root_path = datasource["root_path"]
importer = stp.io.get_method(datasource["importer"], "importer")
for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")
    print('Analyzing event start , end', curdate,',',enddate)

    while curdate <= enddate:
        print('Analyzing', curdate)
        ## read two consecutive radar fields
        fns = stp.io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                              datasource["fn_pattern"], datasource["fn_ext"],
                                              datasource["timestep"], num_prev_files=num_prev_files)
        R,_,metadata = stp.io.readers.read_timeseries(fns, importer, **datasource["importer_kwargs"])

        ## convert to mm/h
        R, metadata = stp.utils.to_rainrate(R, metadata)

        ## threshold the data
        R[R<0.1] = 0.0
        metadata["threshold"] = 0.1

        ## set NaN equal to zero
        R[~np.isfinite(R)] = 0.0

        ## copy the original data
        R_ = R.copy()

        ## set NaN equal to zero
        R_[~np.isfinite(R_)] = 0.0

        ## transform to dBR
        R_, metadata_dbr = stp.utils.dB_transform(R_, metadata, zerovalue=zero_value_dbr)

        ## Compute motion field
        oflow_method = stp.motion.get_method(of_method)
        if of_method == "darts":
            UV = oflow_method(R_)
        else:
            UV = oflow_method(R_[-2:,:,:])

        # Compute different type of nowcasts
        stochastic_nowcast_exists = False
        for nowcast in nowcast_types:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print("Running experiment:", nowcast)
            # Stochastic nowcast
            if (nowcast == "STOCH") or (nowcast == "MEAN"):
                if stochastic_nowcast_exists == False:
                    R_fct = nc(R_[-3:, :, :], UV, n_lead_times, R_thr=metadata_dbr["threshold"], extrap_method=extrap_method, ar_order=ar_order, num_workers=4,
                                kmperpixel=1.0, timestep=datasource["timestep"], n_ens_members=n_members, probmatching_method=pmatching_method, mask_kwargs={'mask_rim':10},
                                n_cascade_levels=n_levels, noise_stddev_adj=noise_stddev_adj, mask_method=mask_method, bandpass_filter_method=bandpass_filter,
                                conditional=conditional_stats, vel_pert_method=None, noise_method=filter, fft_method="numpy", seed=seed)
                    stochastic_nowcast_exists = True
                # Ensemble mean
                if (nowcast == 'MEAN'):
                    if stochastic_nowcast_exists == False:
                        R_fct = nc(R_[-3:, :, :], UV, n_lead_times, R_thr=metadata_dbr["threshold"], extrap_method=extrap_method, ar_order=ar_order, num_workers=4,
                                    kmperpixel=1.0, timestep=datasource["timestep"], n_ens_members=n_members, probmatching_method=pmatching_method, mask_kwargs={'mask_rim':10},
                                    n_cascade_levels=n_levels, noise_stddev_adj=noise_stddev_adj, mask_method=mask_method, bandpass_filter_method=bandpass_filter,
                                    conditional=conditional_stats, vel_pert_method=None, noise_method=filter, fft_method="numpy", seed=seed)
                        stochastic_nowcast_exists = True

                    # Replace nans and set zeros
                    R_fct[np.isnan(R_fct)] = metadata_dbr["zerovalue"]
                    # Back to rainrate
                    R_fct_rate, metadata_dbr = stp.utils.dB_transform(R_fct, metadata_dbr, inverse=True)
                    # Ensemble mean
                    R_fct_mean = stp.postprocessing.ensemblestats.mean(R_fct_rate)
                    # Probability matching the ensemble mean
                    if pmatching_method is not None:
                        for t in range(0, R_fct_mean.shape[0]):
                            R_fct_mean[t,:,:] = stp.postprocessing.probmatching.nonparam_match_empirical_cdf(R_fct_mean[t,:,:], R[-1,:,:])
                    # To dBR again
                    R_fct, metadata_dbr = stp.utils.dB_transform(R_fct_mean, metadata_dbr)
                    R_fct = R_fct[np.newaxis,:]
            elif nowcast == "SPROG":
                # SPROG nowcast
                R_fct = nc(R_[-3:, :, :], UV, n_lead_times, n_ens_members=1, R_thr=metadata_dbr["threshold"], extrap_method=extrap_method, ar_order=ar_order,
                            kmperpixel=1.0, timestep=datasource["timestep"], probmatching_method=pmatching_method, mask_kwargs={'mask_rim':10},
                            n_cascade_levels=n_levels, noise_stddev_adj=noise_stddev_adj, mask_method=mask_method, bandpass_filter_method=bandpass_filter,
                            conditional=conditional_stats, vel_pert_method=None, noise_method=None, fft_method="numpy")

            # Option to animate data to check that forecast fields look alright
            if animate:
                R_fct_, metadata_ = stp.utils.dB_transform(R_fct, metadata_dbr, inverse=True)

                stp.plt.animate(R, nloops=nloops, timestamps=metadata["timestamps"],
                R_fct=R_fct_, timestep_min=datasource["timestep"], UV=UV,
                motion_plot=stp.rcparams.plot.motion_plot, step=60,
                geodata=metadata, map="cartopy", fig_dpi=150,
                colorscale=stp.rcparams.plot.colorscale,
                type="ensemble", prob_thr=1.0,
                plotanimation=True, savefig=True,
                path_outputs="figures", axis="off")
                
            # Replace nans and set zeros
            R_fct[np.isnan(R_fct)] = metadata_dbr["zerovalue"]
            R_fct[R_fct < metadata_dbr["threshold"]] = metadata_dbr["zerovalue"]

            # Plot Fourier spectrum of observations
            plt.figure()
            ax = plt.subplot(111)

            # Remove rain/no-rain discontinuity so that dBR field starts from 0
            R_obs_shift = stp.utils.remove_rain_norain_discontinuity(R_[-1,:,:])
            # Apply window function to reduce edge effects when rain touches the borders
            window = build_2D_tapering_function(R_obs_shift.shape, win_type='flat-hanning')
            R_obs_shift *= window
            # Compute and plot RAPSD
            R_obs_spectrum, fft_freq = stp.utils.rapsd(R_obs_shift, np.fft, d=1.0, return_freq=True)
            stp.plt.plot_spectrum1d(fft_freq, R_obs_spectrum, x_units='km', y_units='dBR', label='Observations', wavelength_ticks=wavelength_ticks, color='k', lw=1.0, ax=ax)

            # Plot Fourier spectra of forecasts
            colors=iter(cm.Blues_r(np.linspace(0,1,len(plot_leadtimes)+2)))
            for t in plot_leadtimes:
                if nowcast == 'STOCH':
                    # Take average spectrum of ensemble members
                    for m in range(0, n_members):
                        R_fct_shift = stp.utils.remove_rain_norain_discontinuity(R_fct[m,t,:,:])
                        R_fct_shift *= window
                        if m == 0:
                            R_fct_spectrum, fft_freq = stp.utils.rapsd(R_fct_shift, np.fft, d=1.0, return_freq=True)
                        else:
                            R_fct_spectrum += stp.utils.rapsd(R_fct_shift, np.fft, d=1.0)
                    # Compute average spectrum
                    R_fct_spectrum/=n_members
                else:
                    R_fct_shift = stp.utils.remove_rain_norain_discontinuity(R_fct[0,t,:,:])
                    R_fct_shift *= window
                    R_fct_spectrum, fft_freq = stp.utils.rapsd(R_fct_shift, np.fft, d=1.0, return_freq=True)

                # Plot RAPSD
                stp.plt.plot_spectrum1d(fft_freq, R_fct_spectrum, color=next(colors), lw=1.0, ax=ax, label='+' + str(int((t+1)*metadata["accutime"])) + ' min')

            # Decorate plot
            plt.legend(loc="lower left")
            if n_levels == 1:
                str_levels = ' - %i level' % n_levels
            else:
                str_levels = ' - %i levels' % n_levels
            # Title
            if nowcast == "STOCH":
                ax.set_title('(a) Ensemble members', fontsize=18)
            if nowcast == "MEAN":
                if data_source[0:3] == "fmi":
                    ax.set_title('(b) Ensemble mean', fontsize=18)
                else:
                    if n_levels == 8:
                        ax.set_title('(a) Ensemble mean', fontsize=18)
                    if n_levels == 1:
                        ax.set_title('(b) Ensemble mean', fontsize=18)
            if nowcast == "SPROG":
                ax.set_title('(c) S-PROG', fontsize=18)
            
            ax.set_ylim([-30,60])
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.grid()

            # Savefig
            if nowcast == 'SPROG':
                figname = 'figures/%s_%s_%s_nlevels%i_spectra.%s' % (data_source[0:3], curdate.strftime("%Y%m%d%H%M"),nowcast,n_levels,fmt)
            else:
                figname = 'figures/%s_%s_%s_nmembers%i_nlevels%i_spectra.%s' % (data_source[0:3], curdate.strftime("%Y%m%d%H%M"),nowcast,n_members,n_levels,fmt)
            plt.savefig(figname, bbox_inches="tight", dpi=200)
            print(figname, 'saved.')

        curdate += timedelta(minutes=timestep_run)

print("Finished!")