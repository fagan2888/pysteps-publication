#!/bin/env python

"""
Cascade experiment:
Pysteps ensembles are generated and verified with/without cascade decomposition and with/without precipitation mask.
Verification results are saved in a binary file.
"""

import dask
from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import sys
import time
import sys

from pysteps import rcparams, io, motion, nowcasts, utils
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import transformation, conversion
from pysteps.verification import ensscores
from pysteps.verification import probscores
from pysteps.utils import aggregate_fields_space, clip_domain

# Data
sys.path.insert(0,'..')
import precipevents

data_source = "mch_hdf5"
time_res = 5
if data_source == "fmi":
    # all events
    precipevents = precipevents.fmi
    # only one event (comment out if you want all events)
    precipevents = [("201609291000", "201609291800")]
if data_source == "mch_hdf5":
    # all events
    precipevents = precipevents.mch
    # only 1 event (comment out if you want all events)
    # precipevents = [("201701311000", "201701311000")]

# Cascade experiments
cascade_levels = [1,8]
mask_methods = ['incremental', None]

# Forecast parameters
timestep = 30
R_min = 0.1

num_timesteps = 12
ensemble_size = 24
num_workers = 12
seed = 24

# Output file basename containing the verification statistics
filename_verif_base = "data/" + data_source[0:3] + "_cascade_results"

# Verification parameters
R_thrs = [0.1, 1.0, 5.0]            # Rainfall thresholds to verify (same applies for accumulations)
v_scales_km = [1, 10, 40]           # Spatial scales to verify [km] (must be divisors of clipped grid size)
v_accu_min = 5                     # Temporal accumulation to verify [min]. Only one value possible

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# 
# Lead times to verify [min]
if v_accu_min == 60:
    v_leadtimes = [60]             
if v_accu_min == 30:
    v_leadtimes = [30,60]    
if v_accu_min == time_res:
    v_leadtimes = [5,10,15,20,25,30,45,60]
    
# Derive indices for lead times 
v_leadtimes = np.array(v_leadtimes)
idx_leadtimes = (np.array(v_leadtimes)/v_accu_min - 1).astype(int)
print("Verifying lead times [min]:", v_leadtimes, ", indices:", idx_leadtimes)  
 
# Check if verification parameters are coherent    
if v_leadtimes.max() >  num_timesteps*time_res:
    print("num_timesteps should be large enough to contain the maximum v_leadtimes")
    sys.exit()
if v_leadtimes.min() < v_accu_min:
    print("Minimum valid lead time in v_leadtimes should larger than the accumulation time")
    sys.exit()
filename_verif = filename_verif_base + "_accum" + ("%02i" % v_accu_min) + ".dat"
v_leadtimes = v_leadtimes.tolist()

# vp_par  = (2.56338484, 0.3330941, -2.99714349) # mch only
# vp_perp = (1.31204508, 0.3578426, -1.02499891)
vp_par  = (2.31970635, 0.33734287, -2.64972861) # fmi+mch
vp_perp = (1.90769947, 0.33446594, -2.06603662)

datasource = rcparams.data_sources[data_source]
root_path = datasource["root_path"]
importer = io.get_method(datasource["importer"], "importer")

# read Swiss radar data quality (list of timestamps with low-quality observations)
badts2016 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2016_radar-stats-badTimestamps_00005.txt", dtype="str")
badts2017 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2017_radar-stats-badTimestamps_00005.txt", dtype="str")
badts = np.concatenate((badts2016, badts2017))

# Instantiate dictionary containing the verification results
results = {}
results["metadata"] = {}
results["metadata"]["v_accu_min"] = v_accu_min
results["metadata"]["v_leadtimes"] = v_leadtimes
results["metadata"]["v_scales_km"] = v_scales_km

for c in cascade_levels:
    for m in mask_methods:
        results[c,m] = {}
        results[c,m]["reldiag"] = {}
        results[c,m]["rankhist"] = {}
        results[c,m]["ROC"] = {}
        for R_thr in R_thrs:
            results[c,m]["reldiag"][R_thr] = {}
            results[c,m]["rankhist"][R_thr] = {}
            results[c,m]["ROC"][R_thr] = {}
            for scale_km in v_scales_km:
                results[c,m]["reldiag"][R_thr][scale_km] = {}
                results[c,m]["rankhist"][R_thr][scale_km] = {}
                results[c,m]["ROC"][R_thr][scale_km] = {}
                for lt in v_leadtimes:
                    results[c,m]["reldiag"][R_thr][scale_km][lt] = probscores.reldiag_init(R_thr, n_bins=10)
                    results[c,m]["rankhist"][R_thr][scale_km][lt] = ensscores.rankhist_init(ensemble_size, R_thr)
                    results[c,m]["ROC"][R_thr][scale_km][lt] = probscores.ROC_curve_init(R_thr, n_prob_thrs=100)

for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")
    
    countnwc = 0
    while curdate <= enddate:

        if curdate + num_timesteps*timedelta(minutes=datasource["timestep"]) > enddate:
            break
        countnwc += 1
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Computing nowcast %d for event %d, start date %s..." % (countnwc, pei+1, str(curdate)))
        
        # Get observations to start nowcast
        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=3)

        R,_,metadata = io.readers.read_timeseries(fns, importer, **datasource["importer_kwargs"])
                                                  
        missing_data = False
        for i in range(R.shape[0]):
            if not np.any(np.isfinite(R[i, :, :])):
                print("Skipping, no finite values found for time step %d" % (i+1))
                missing_data = True
                break
                
        low_quality = False
        for timestamp in metadata["timestamps"]:
            if timestamp.strftime("%Y%m%d%H%M") in badts:
                print("Skipping, low quality observation found for time step %d" % (i+1))
                low_quality = True
                break

        if missing_data or low_quality:
            curdate += timedelta(minutes=timestep)
            continue

        R[~np.isfinite(R)] = metadata["zerovalue"]
        R, metadata = conversion.to_rainrate(R, metadata)
        R, metadata = transformation.dB_transform(R, metadata, threshold=R_min)
        
        # Get observations for verification
        obs_fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                          datasource["fn_pattern"], datasource["fn_ext"],
                                          datasource["timestep"], 
                                          num_next_files=num_timesteps)
        obs_fns = (obs_fns[0][1:], obs_fns[1][1:])
        if len(obs_fns[0]) == 0:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        R_obs,_,metaobs = io.readers.read_timeseries(obs_fns, importer, **datasource["importer_kwargs"]) 
        R_obs, metaobs = conversion.to_rainrate(R_obs, metaobs)
        R_obs[R_obs < R_min] = 0.0
        metaobs["threshold"] = R_min
        
        # Compute observed accumulations
        metafct = metaobs.copy()
        if v_accu_min > time_res:
            aggregator = utils.get_method("accumulate")
            R_obs, metaobs = aggregator(R_obs, metaobs, v_accu_min)

            R_obs[R_obs < R_min] = 0.0
            metaobs["threshold"] = R_min            
        
        if R_obs is None:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        for c in cascade_levels:
            print('----------------------------------')
            if c == 1:
                bandpass_filter = "uniform"
            else:
                bandpass_filter = "gaussian"
            for m in mask_methods:
                tic = time.time()
                
                # Compute optical flow
                oflow = motion.get_method("lucaskanade")
                V = oflow(R)
                
                # Compute nowcast
                nc = nowcasts.steps.forecast
                vel_pert_kwargs = {"p_pert_par":vp_par , "p_pert_perp":vp_perp}
                R_fct = nc(R[-3:, :, :], V, num_timesteps, R_thr=metadata["threshold"], 
                            kmperpixel=1.0, timestep=5, n_ens_members=ensemble_size, 
                            n_cascade_levels=c, mask_method=m, bandpass_filter_method=bandpass_filter,
                            vel_pert_method="bps", vel_pert_kwargs=vel_pert_kwargs,
                            num_workers=num_workers, fft_method="numpy", seed=seed)
                
                # Transform back to rainrates
                R_fct = transformation.dB_transform(R_fct, metadata, inverse=True)[0]
                
                # Compute forecast accumulations
                if v_accu_min > time_res:
                    R_fct, metafct_accum = aggregator(R_fct, metafct, v_accu_min)
                    
                    R_fct[R_fct < R_min] = 0.0
                    metafct_accum["threshold"] = R_min
                
                # Clip domain
                if domain[0:3] == 'mch':
                    xlim = [400000, 840000]
                    ylim = [-50000, 350000]
                    extent = (xlim[0], xlim[1], ylim[0], ylim[1]) 
                    R_fct_c, metafct_c = clip_domain(R_fct, metafct, extent)
                    R_obs_c, metaobs_c = clip_domain(R_obs, metaobs, extent)
                else:
                    print("To verify upscaled fields you should clip domain appropriately")
                    sys.exit()
                
                # Verify nowcasts
                print("Verifying nowcasts...")
                def worker(lt, R_thr, scale_km):
                    lt_idx = int(lt/v_accu_min - 1)
                    if not np.any(np.isfinite(R_obs_c[lt_idx, :, :])):
                        return 
                    # print(c,m,R_thr,scale_km,lt,lt_idx)                    
                    R_fct_s,_ = aggregate_fields_space(R_fct_c, metafct_c, scale_km*1000, ignore_nan=False)
                    R_obs_s,_ = aggregate_fields_space(R_obs_c, metaobs_c, scale_km*1000, ignore_nan=False)
                    
                    P_fct = excprob(R_fct_s[:, lt_idx, :, :], R_thr, ignore_nan=True)
                    
                    probscores.reldiag_accum(results[c,m]["reldiag"][R_thr][scale_km][lt], 
                                             P_fct, R_obs_s[lt_idx, :, :])
                    ensscores.rankhist_accum(results[c,m]["rankhist"][R_thr][scale_km][lt], 
                                             R_fct_s[:, lt_idx, :, :], R_obs_s[lt_idx, :, :])
                    probscores.ROC_curve_accum(results[c,m]["ROC"][R_thr][scale_km][lt], 
                                               P_fct, R_obs_s[lt_idx, :, :])
                res = []
                for R_thr in R_thrs:
                    for scale_km in v_scales_km:
                        for lt in v_leadtimes:
                            lt_idx = int(lt/v_accu_min - 1)
                            missing_data = np.all(~np.isfinite(R_obs[lt_idx, :, :]))
                            low_quality = metaobs["timestamps"][lt_idx].strftime("%Y%m%d%H%M") in badts
                            if missing_data or low_quality:
                                print("Warning: no verifying observations for lead time %d." % (lt))
                                continue
                            res.append(dask.delayed(worker)(lt, R_thr, scale_km))
                dask.compute(*res, num_workers=num_workers)
                
                toc = time.time()
                print('Elapsed time:', toc-tic)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        with open(filename_verif, "wb") as f:
            pickle.dump(results, f)

        curdate += timedelta(minutes=timestep)

with open(filename_verif, "wb") as f:
    pickle.dump(results, f)
print(filename_verif, 'written.')