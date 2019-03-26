'''
Script to analyze the temporal auto-correlation of stochastic ensemble members 
and observations in Lagrangian frame at different spatial scales. The observations
are analyzed both at nowcast start time and during the nowcast to understand the
impact of non-stationary motion.
'''
from collections import defaultdict
from datetime import datetime, timedelta
import os
import pickle
import sys
import numpy as np
from pysteps import rcparams, cascade, extrapolation, io, motion, utils, nowcasts
from pysteps.timeseries import autoregression
from pysteps.utils import remove_rain_norain_discontinuity
import matplotlib.pyplot as plt

sys.path.insert(0,'..')
import precipevents

# the domain: "fmi" or "mch_hdf5"
domain = "mch_hdf5"
timestep = 240

# Experiment parameters
n_levels        = [1,8]   # to produce the nowcast [1 or 8]
n_levels_verif  = 8       # to verify the nowcast [keep it fixed]
recompute_flow  = False   # whether to recompute the flow of the fct/obs fields (very slow)

nhours_ar       = 2000    # max number of hours to integrate the full ACF
min_rho_level0  = 0.95    # minimum auto-correlation of level 0 to consider as valid

# Forecast parameters
n_lead_times        = 12
n_members           = 1
ar_order            = 2
oflow_method        = "lucaskanade"
filter              = "nonparametric"
mask_method         = "incremental"
pmatching_method    = "cdf"
zero_value_dbr      = -15
noise_stddev_adj    = None
conditional_stats   = False
seed                = None

filename_out = "data/%s_ar2_corr_results_recomputeflow-%s.dat" % (domain,recompute_flow)
if domain == "fmi":
    precipevents = precipevents.fmi
    # only one event (comment out if you want all events)
    precipevents = [("201609281500", "201609281800")]
if domain[0:3] == "mch":
    precipevents = precipevents.mch
    # only 1 event (comment out if you want all events)
    precipevents = [("201701311000", "201701311400")]

datasource = rcparams.data_sources[domain]
root_path = datasource["root_path"]
importer = io.get_method(datasource["importer"], "importer")

oflow = motion.get_method(oflow_method)
extrapolator = extrapolation.get_method("semilagrangian")
nc = nowcasts.steps.forecast

filter_verif = None   

# Dictionary containing the sum of temporal autocorrelation functions
results = {}
# max number of steps to integrate the full ACF
nsteps_ar = int(nhours_ar*(60/datasource["timestep"]))
for lev in n_levels:
    results[lev] = {}

    results[lev]["leadtimes"] = (np.arange(nsteps_ar)+1)*datasource["timestep"]
    results[lev]["timestep"] = datasource["timestep"]

    results[lev]["cc_fct"] = [np.zeros(nsteps_ar) for i in range(n_levels_verif)] 
    results[lev]["cc_obs"] = [np.zeros(nsteps_ar) for i in range(n_levels_verif)]
    results[lev]["cc_obs_t0"] = [np.zeros(nsteps_ar) for i in range(n_levels_verif)]
    results[lev]["n_fct_samples"] = [np.zeros(nsteps_ar, dtype=int) for i in range(n_levels_verif)]
    results[lev]["n_obs_samples"] = [np.zeros(nsteps_ar, dtype=int) for i in range(n_levels_verif)]
    results[lev]["n_obs_samples_t0"] = [np.zeros(nsteps_ar, dtype=int) for i in range(n_levels_verif)]

for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")
    
    while curdate <= enddate:
        print("Running analyses for event %d, start date %s..." % (pei+1, str(curdate)), end="")
        sys.stdout.flush()

        if curdate + n_lead_times * timedelta(minutes=5) > enddate:
            print("Done.")
            break
        
        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=2)

        R,_,metadata = io.readers.read_timeseries(fns, importer,
                                                  **datasource["importer_kwargs"])

        missing_data = False
        for i in range(R.shape[0]):
            if not np.any(np.isfinite(R[i, :, :])):
                print("Skipping, no finite values found for time step %d" % (i+1))
                missing_data = True
                break
        
        # get radar mask
        mask = (~np.isnan(R[-1,:,:])).astype(float)
        
        # convert to mm/h
        R, metadata = utils.to_rainrate(R, metadata)

        # threshold the data
        R[R < 0.1] = 0.0
        metadata["threshold"] = 0.1

        # set NaN equal to zero
        R[~np.isfinite(R)] = 0.0

        # transform to dBR
        R, metadata = utils.dB_transform(R, metadata, zerovalue=zero_value_dbr)
        
        R_ = R.copy()
            
        # Remove rain/no-rain discontinuity so that dBR field starts from 0
        R = remove_rain_norain_discontinuity(R)
            
        # Read verifying observations
        obs_fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                          datasource["fn_pattern"], datasource["fn_ext"],
                                          datasource["timestep"], num_next_files=n_lead_times)
        R_obs,_,metadata_obs = io.read_timeseries(obs_fns, importer, **datasource["importer_kwargs"])
        
        # convert to mm/h
        R_obs, metadata_obs = utils.to_rainrate(R_obs, metadata_obs)

        # threshold the data
        R_obs[R_obs < 0.1] = 0.0
        metadata_obs["threshold"] = 0.1

        # set NaN equal to zero
        R_obs[~np.isfinite(R_obs)] = 0.0

        # transform to dBR
        R_obs, metadata_obs = utils.dB_transform(R_obs, metadata_obs, zerovalue=zero_value_dbr)
        
        R_obs_ = R_obs.copy()
        
        # Remove rain/no-rain discontinuity so that dBR field starts from 0
        R_obs = remove_rain_norain_discontinuity(R_obs)
            
        ## Compute motion field
        if oflow_method == "darts":
            UV = oflow(R_)
        else:
            UV = oflow(R_[-2:,:,:])
        
        for lev in n_levels:
            bandpass_filter = 'uniform' if (lev == 1) else 'gaussian'
            
            ## Compute stochastic nowcast
            print("Computing nowcasting with", lev, "levels...")
            R_fct = nc(R_[-3:, :, :], UV, n_lead_times, R_thr=metadata["threshold"], ar_order=ar_order, num_workers=4,
                                    kmperpixel=1.0, timestep=datasource["timestep"], n_ens_members=n_members, probmatching_method=pmatching_method, mask_kwargs={'mask_rim':10},
                                    n_cascade_levels=lev, noise_stddev_adj=noise_stddev_adj, mask_method=mask_method, bandpass_filter_method=bandpass_filter,
                                    conditional=conditional_stats, vel_pert_method=None, noise_method=filter, fft_method="numpy", seed=seed)
            # Replace nans and set zeros
            R_fct[np.isnan(R_fct)] = metadata["zerovalue"]
            R_fct[R_fct < metadata["threshold"]] = metadata["zerovalue"]
            
            # Remove rain/no-rain discontinuity so that dBR field starts from 0
            R_fct = remove_rain_norain_discontinuity(R_fct)
            
            ## Derive AR-2 ACF from forecast sequences
            print('-------------------------------------')
            print("Computing ACF of nowcasts...")
            if filter_verif is None:
                if n_levels_verif == 1:
                    filter_verif = cascade.bandpass_filters.filter_uniform(R.shape[1:], n_levels_verif)
                else:
                    filter_verif = cascade.bandpass_filters.filter_gaussian(R.shape[1:], n_levels_verif, d=metadata["xpixelsize"]/1000.0) 
                    central_wavelengths_km = 1.0/filter_verif["central_freqs"]
                results[lev]["central_wavelengths"] = central_wavelengths_km
                
            n_steps = n_lead_times-ar_order
            for lt in range(0, n_steps):
                print('Lead time:', lt)
                for m in range(0, n_members):
                    # Re-compute motion from the three forecast images
                    if recompute_flow:
                        UV = oflow(R_fct[m,lt:lt+2,:,:])
                    
                    # Put in Lagrangian coordinates the three forecast images
                    R_minus_2 = extrapolation.semilagrangian.extrapolate(R_fct[m,lt, :, :], UV, 2, outval=zero_value_dbr)[-1, :, :]
                    R_minus_1 = extrapolation.semilagrangian.extrapolate(R_fct[m,lt+1, :, :], UV, 1, outval=zero_value_dbr)[-1, :, :]
                    
                    # Cascade decomposition
                    c1 = cascade.decomposition.decomposition_fft(R_minus_2, filter_verif)
                    c2 = cascade.decomposition.decomposition_fft(R_minus_1, filter_verif)
                    c3 = cascade.decomposition.decomposition_fft(R_fct[m,lt+2, :, :], filter_verif)
                    
                    # Compute autocorrelation coefficients and function at each level
                    for i in range(n_levels_verif):
                        gamma_1 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                              c2["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                        gamma_2 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                              c1["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                        
                        if i == 0 and gamma_1 < min_rho_level0:
                            print("Too low correlation at level 0, gamma_1 =", gamma_1)
                            break
                            
                        gamma_2 = autoregression.adjust_lag2_corrcoef2(gamma_1, gamma_2)
                        acf = autoregression.ar_acf([gamma_1,gamma_2], n=nsteps_ar)
                        
                        results[lev]["cc_fct"][i] += np.array(acf)
                        results[lev]["n_fct_samples"][i] += 1
            
            ## Derive AR-2 ACF from observed sequences (for each lead time)
            print("Computing ACF of observations at each lead time...")        
            for lt in range(0, n_steps):
                print('Lead time:', lt)
                # Re-compute motion from the three observed images
                if recompute_flow:
                    UV = oflow(R_obs_[lt:lt+2,:,:])
                
                # Put in Lagrangian coordinates the three observed images
                R_minus_2 = extrapolation.semilagrangian.extrapolate(R_obs[lt, :, :], UV, 2, outval=zero_value_dbr)[-1, :, :]
                R_minus_1 = extrapolation.semilagrangian.extrapolate(R_obs[lt+1, :, :], UV, 1, outval=zero_value_dbr)[-1, :, :]
                
                # Cascade decomposition
                c1 = cascade.decomposition.decomposition_fft(R_minus_2, filter_verif)
                c2 = cascade.decomposition.decomposition_fft(R_minus_1, filter_verif)
                c3 = cascade.decomposition.decomposition_fft(R_obs[lt+2, :, :], filter_verif)
                
                # Compute autocorrelation coefficients
                for i in range(n_levels_verif):
                    gamma_1 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                          c2["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                    gamma_2 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                          c1["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                    
                    if i == 0 and gamma_1 < min_rho_level0:
                        print("Too low correlation at level 0, gamma_1 =", gamma_1)
                        break
                            
                    gamma_2 = autoregression.adjust_lag2_corrcoef2(gamma_1, gamma_2)
                    acf_obs = autoregression.ar_acf([gamma_1, gamma_2],n=nsteps_ar)
                    
                    results[lev]["cc_obs"][i] += np.array(acf_obs)
                    results[lev]["n_obs_samples"][i] += 1
            
            ## Derive AR-2 ACF from observed sequence (only start time)
            print("Computing ACF of observations at start time...")
            
            # Put in Lagrangian coordinates the three observed images
            R_minus_2 = extrapolation.semilagrangian.extrapolate(R[-3, :, :], UV, 2, outval=zero_value_dbr)[-1, :, :]
            R_minus_1 = extrapolation.semilagrangian.extrapolate(R[-2, :, :], UV, 1, outval=zero_value_dbr)[-1, :, :]
            
            # Cascade decomposition
            c1 = cascade.decomposition.decomposition_fft(R_minus_2, filter_verif)
            c2 = cascade.decomposition.decomposition_fft(R_minus_1, filter_verif)
            c3 = cascade.decomposition.decomposition_fft(R[-1, :, :], filter_verif)
            
            # Compute autocorrelation coefficients
            for i in range(n_levels_verif):
                gamma_1 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                      c2["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                gamma_2 = np.corrcoef(c3["cascade_levels"][i, :, :][mask==1].flatten(),
                                      c1["cascade_levels"][i, :, :][mask==1].flatten())[0, 1]
                
                if i == 0 and gamma_1 < min_rho_level0:
                    print("Too low correlation at level 0, gamma_1 =", gamma_1)
                    break
                
                gamma_2 = autoregression.adjust_lag2_corrcoef2(gamma_1, gamma_2)
                acf_obs = autoregression.ar_acf([gamma_1, gamma_2],n=nsteps_ar)
                
                results[lev]["cc_obs_t0"][i] += np.array(acf_obs)
                results[lev]["n_obs_samples_t0"][i] += 1
                    
            print("Done.")

        curdate += timedelta(minutes=timestep)
        with open(filename_out, "wb") as f:
            pickle.dump(results, f)

with open(filename_out, "wb") as f:
    pickle.dump(results, f)
print(filename_out, "saved.")