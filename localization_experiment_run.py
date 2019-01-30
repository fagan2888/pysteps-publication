"""
Section 6.3
This experiment investigates the impact of localization on the quality  of  the  
nowcast.  For  localization  we  intend  the  use of  a  subset  of  the  
observations  in  order  to  estimate  model parameters that are distributed in 
space. The short-space approach used in Nerini et al. (2017) is generalized to 
the whole nowcasting system. This essenially boils down to a moving window 
localization of the nowcasting procedure, whereby all parameters are estimated 
over a subdomain of prescribed size.

This script runs the nowcasts and produces the verification results for a set
of localization windows.
"""
import dask
from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import sys
import time
from pysteps import io, motion, nowcasts, utils
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import transformation, conversion
from pysteps.verification import ensscores
from pysteps.verification import probscores
import datasources, precipevents

window_sizes = [710, 360, 180, 90]
domain = "mch_rzc"
timestep = 30
num_workers = 24
num_timesteps = 18
ensemble_size = 24
R_min = 0.1
R_thrs = [0.1, 1.0, 5.0, 10.0]

datasource = datasources.mch_rzc
precipevents = precipevents.mch

vp_par  = (2.31970635, 0.33734287, -2.64972861) # fmi+mch
vp_perp = (1.90769947, 0.33446594, -2.06603662)

root_path = os.path.join(datasources.root_path, datasource["root_path"])
importer = io.get_method(datasource["importer"], "importer")

# read Swiss radar data quality (list of timestamps with low-quality observations)
badts2016 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2016_radar-stats-badTimestamps_00005.txt", dtype="str")
badts2017 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2017_radar-stats-badTimestamps_00005.txt", dtype="str")
badts = np.concatenate((badts2016, badts2017))

results = {}

for ws in window_sizes:
    results[ws] = {}
    results[ws]["reldiag"] = {}
    results[ws]["rankhist"] = {}
    results[ws]["ROC"] = {}
    for R_thr in R_thrs:
        results[ws]["reldiag"][R_thr] = {}
        results[ws]["rankhist"][R_thr] = {}
        results[ws]["ROC"][R_thr] = {}
        for lt in range(num_timesteps):
            results[ws]["reldiag"][R_thr][lt] = probscores.reldiag_init(R_thr, n_bins=10)
            results[ws]["rankhist"][R_thr][lt] = ensscores.rankhist_init(ensemble_size, R_thr)
            results[ws]["ROC"][R_thr][lt] = probscores.ROC_curve_init(R_thr, n_prob_thrs=100)

for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")
    
    countnwc = 0
    while curdate <= enddate:

        if curdate + num_timesteps*timedelta(minutes=datasource["timestep"]) > enddate:
            break
        countnwc += 1    
        print("Computing nowcast %d for event %d, start date %s..." % (countnwc, pei+1, str(curdate)))

        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=3)

        R,_,metadata = io.readers.read_timeseries(fns, importer,
                                                  **datasource["importer_kwargs"])

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

        obs_fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                          datasource["fn_pattern"], datasource["fn_ext"],
                                          datasource["timestep"], 
                                          num_next_files=num_timesteps)
        obs_fns = (obs_fns[0][1:], obs_fns[1][1:])
        if len(obs_fns[0]) == 0:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        R_obs,_,metaobs = io.readers.read_timeseries(obs_fns, importer,
                                                 **datasource["importer_kwargs"])
        R_obs, metaobs = conversion.to_rainrate(R_obs, metaobs)

        if R_obs is None:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        for ws in window_sizes:
            oflow = motion.get_method("lucaskanade")
            V = oflow(R)
            nc = nowcasts.sseps.forecast
            vel_pert_kwargs = {"p_pert_par":vp_par , "p_pert_perp":vp_perp}
            R_fct = nc(R[-3:, :, :], metadata, V, num_timesteps, 
                       n_ens_members=ensemble_size, n_cascade_levels=6, 
                       win_size=ws, overlap=0.1,
                       vel_pert_method="bps", 
                       mask_method="incremental", num_workers=num_workers, 
                       fft_method="numpy", vel_pert_kwargs=vel_pert_kwargs)

            R_fct = transformation.dB_transform(R_fct, metadata, inverse=True)[0]

            def worker(lt, R_thr):
                if not np.any(np.isfinite(R_obs[lt, :, :])):
                    return

                P_fct = excprob(R_fct[:, lt, :, :], R_thr)

                probscores.reldiag_accum(results[ws]["reldiag"][R_thr][lt], 
                                         P_fct, R_obs[lt, :, :])
                ensscores.rankhist_accum(results[ws]["rankhist"][R_thr][lt], 
                                         R_fct[:, lt, :, :], R_obs[lt, :, :])
                probscores.ROC_curve_accum(results[ws]["ROC"][R_thr][lt], 
                                           P_fct, R_obs[lt, :, :])

            res = []
            for lt in range(num_timesteps):
                missing_data = np.all(~np.isfinite(R_obs[lt, :, :]))
                if missing_data:
                    print("Warning: no verifying observations for lead time %d." % (lt+1))
                    continue
                low_quality = metaobs["timestamps"][lt].strftime("%Y%m%d%H%M") in badts
                if low_quality:
                    print("Warning: low quality observations for lead time %d." % (lt+1))
                    continue
                for R_thr in R_thrs:
                    res.append(dask.delayed(worker)(lt, R_thr))
            dask.compute(*res, num_workers=num_workers)

        with open("window_size_results_2.dat", "wb") as f:
            pickle.dump(results, f)

        curdate += timedelta(minutes=timestep)

with open("window_size_results_2.dat", "wb") as f:
    pickle.dump(results, f)
