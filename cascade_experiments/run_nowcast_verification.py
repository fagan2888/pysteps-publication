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

from pysteps import rcparams, io, motion, nowcasts, utils
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import transformation, conversion
from pysteps.verification import ensscores
from pysteps.verification import probscores

# Cascade experiments
cascade_levels = [1,8]
mask_methods = ['incremental', None]

# Output file basename containing the verification statistics
filename_verif_base = "data/cascade_results"

# Verification parameters
R_thrs = [0.1, 1.0, 5.0, 10.0]

verify_accumulation = True
v_leadtimes = [60]
v_accu = 60

# Forecast parameters
domain = "mch_hdf5"
timestep = 30
R_min = 0.1

num_timesteps = 12
ensemble_size = 24
num_workers = 24
seed = 42

if verify_accumulation:
    idx_leadtimes = range(len(v_leadtimes))
    num_timesteps = int(np.max(np.array(v_leadtimes))/5)
    num_lead_times = len(v_leadtimes)
    
    filename_verif = filename_verif_base + "_accum" + ("%02i" % v_accu) + ".dat"
else:
    idx_leadtimes = range(num_timesteps)
    num_lead_times = len(v_leadtimes)
    v_leadtimes = (1+np.array(idx_leadtimes))*v_accu
    v_accu = 5
    
    filename_verif = filename_verif_base + "_rainrate.dat"

# Data
sys.path.insert(0,'..')
import precipevents

datasource = rcparams.data_sources[domain]
precipevents = precipevents.mch
# precipevents = [("201604161800", "201604161900")] # single-event test

# vp_par  = (2.56338484, 0.3330941, -2.99714349) # mch only
# vp_perp = (1.31204508, 0.3578426, -1.02499891)
vp_par  = (2.31970635, 0.33734287, -2.64972861) # fmi+mch
vp_perp = (1.90769947, 0.33446594, -2.06603662)

root_path = datasource["root_path"]
importer = io.get_method(datasource["importer"], "importer")

# read Swiss radar data quality (list of timestamps with low-quality observations)
badts2016 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2016_radar-stats-badTimestamps_00005.txt", dtype="str")
badts2017 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2017_radar-stats-badTimestamps_00005.txt", dtype="str")
badts = np.concatenate((badts2016, badts2017))

# Instantiate dictionary containing the verification results
results = {}
results["metadata"] = {}
results["metadata"]["accumulation"] = verify_accumulation
results["metadata"]["v_accu"] = v_accu
results["metadata"]["v_leadtimes"] = v_leadtimes

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
            for lt in range(num_lead_times):
                results[c,m]["reldiag"][R_thr][lt] = probscores.reldiag_init(R_thr, n_bins=10)
                results[c,m]["rankhist"][R_thr][lt] = ensscores.rankhist_init(ensemble_size, R_thr)
                results[c,m]["ROC"][R_thr][lt] = probscores.ROC_curve_init(R_thr, n_prob_thrs=100)

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
        if verify_accumulation:
            aggregator = utils.get_method("accumulate")
            metafct = metaobs.copy()
            R_obs, metaobs = aggregator(R_obs, metaobs, v_accu)

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
                if verify_accumulation:
                    R_fct, metafct_accum = aggregator(R_fct, metafct, v_accu)
                    
                    R_fct[R_fct < R_min] = 0.0
                    metafct_accum["threshold"] = R_min
                
                # Verify nowcasts
                def worker(lt, R_thr):
                    if not np.any(np.isfinite(R_obs[lt, :, :])):
                        return

                    P_fct = excprob(R_fct[:, lt, :, :], R_thr, ignore_nan=True)

                    probscores.reldiag_accum(results[c,m]["reldiag"][R_thr][lt], 
                                             P_fct, R_obs[lt, :, :])
                    ensscores.rankhist_accum(results[c,m]["rankhist"][R_thr][lt], 
                                             R_fct[:, lt, :, :], R_obs[lt, :, :])
                    probscores.ROC_curve_accum(results[c,m]["ROC"][R_thr][lt], 
                                               P_fct, R_obs[lt, :, :])
                res = []
                for R_thr in R_thrs:
                    for lt in idx_leadtimes:
                        missing_data = np.all(~np.isfinite(R_obs[lt, :, :]))
                        low_quality = metaobs["timestamps"][lt].strftime("%Y%m%d%H%M") in badts
                        if missing_data or low_quality:
                            print("Warning: no verifying observations for lead time %d." % (lt+1))
                            continue
                        res.append(dask.delayed(worker)(lt, R_thr))
                dask.compute(*res, num_workers=num_workers)
                
                toc = time.time()
                print('Elapsed time:', toc-tic)
        with open(filename_verif, "wb") as f:
            pickle.dump(results, f)

        curdate += timedelta(minutes=timestep)

with open(filename_verif, "wb") as f:
    pickle.dump(results, f)
print(filename_verif, 'written.')