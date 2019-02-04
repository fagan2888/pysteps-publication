#!/bin/env python

""" 
Script to verify numerical diffusion of deterministic advection nowcasts.
"""

from datetime import datetime, timedelta
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import numpy as np
import sys

import pysteps as stp
from pysteps import rcparams

# Import or define precip events
sys.path.insert(0,'..')
import precipevents
precipevents = precipevents.mch
precipevents = [("201701311000", "201701311000"),
                ("201607111300", "201607111300")]

# Parameters
domain = "mch_hdf5"
timestep_run = 240
n_lead_times = 24
num_prev_files = 9 # Only applies for darts

of_methods = ["darts", "lucaskanade", "vet"]

# Figure parameters   
plot_leadtimes = [11, 23]
linestyles_leadtimes = ['-', '--']     
colors_of = ['C0', 'C1', 'C2']
wavelength_ticks = [512,256,128,64,32,16,8,4,2]
fmt = 'pdf'

# Loop over events
datasource = rcparams.data_sources[domain]
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

        ## copy the original data
        R_ = R.copy()

        ## set NaN equal to zero
        R_[~np.isfinite(R_)] = 0.0

        ## transform to dBR
        R_, metadata_dbr = stp.utils.dB_transform(R_,metadata)

        # Plot Fourier spectrum of observations
        plt.figure()
        ax = plt.subplot(111)
        R_obs_spectrum, fft_freq = stp.utils.rapsd(stp.utils.remove_rain_norain_discontinuity(R_[-1,:,:]), np.fft, d=1.0, return_freq=True)
        stp.plt.plot_rapsd(fft_freq, R_obs_spectrum, x_units='km', y_units='dBR', label='Observation 0 min', wavelength_ticks=wavelength_ticks, color='k', lw=1.0, ax=ax)
        
        c=0
        for of_method in of_methods:
            # Compute motion field
            oflow_method = stp.motion.get_method(of_method)
            if of_method == "darts":
                UV = oflow_method(R_)              
            else:
                UV = oflow_method(R_[-2:,:,:]) 
            
            # Simple advection nowcast
            adv_method = stp.extrapolation.get_method("semilagrangian") 
            R_fct = adv_method(R_[-1,:,:], UV, n_lead_times, verbose=True)
            R_fct[np.isnan(R_fct)] = metadata_dbr["zerovalue"]

            # Plot Fourier spectra of nowcasts
            if of_method == "darts":
                of_method_txt = "DARTS" 
            if of_method == "lucaskanade":
                of_method_txt = "Lucas-Kanade" 
            if of_method == "vet":
                of_method_txt = "VET" 
            l = 0
            for t in range(0,n_lead_times):
                if t in plot_leadtimes:
                    R_fct_shift = stp.utils.remove_rain_norain_discontinuity(R_fct[t,:,:])
                    R_fct_spectrum, fft_freq = stp.utils.rapsd(R_fct_shift, np.fft, d=1.0, return_freq=True)
                    stp.plt.plot_rapsd(fft_freq, R_fct_spectrum, color=colors_of[c], linestyle=linestyles_leadtimes[l], lw=1.0, ax=ax, label=of_method_txt + ' +' + str(int((t+1)*metadata["accutime"])) + ' min')
                    l+=1
            c+=1
           
        # Decorate plot
        plt.legend(loc="lower left")
        
        # ax.set_title(str(curdate), fontsize=16)
        ax.set_ylim([-10,60])
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.grid()
        
        # Savefig
        figname = 'figures/' + curdate.strftime("%Y%m%d%H%M") + '_numerical_diffusion_spectra.' + fmt
        plt.savefig(figname, bbox_inches="tight", dpi=200)
        print(figname, 'saved.')
        
        curdate += timedelta(minutes=timestep_run)
        
print("Finished!")        
