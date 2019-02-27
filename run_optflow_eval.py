# Runs advection-based nowcasts with different optical flow methods (Figure
# 11 in the paper).

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import sys
import time
from pysteps import io, motion, nowcasts, utils
from pysteps.utils import transformation
from pysteps.verification.detcatscores import det_cat_fcst
from pysteps.verification.detcontscores import det_cont_fcst
import datasources, precipevents

# the nowcast method to use: "advection" or "sprog"
# you need to run this script with each to produce the results for Figure 11
#nowcast_method = "advection"
nowcast_method = "sprog"
# the optical flow methods to use
oflow_methods = ["darts", "lucaskanade", "vet"]
# the domain: "fmi" or "mch"
#domain = "fmi"
domain = "mch"
# time step between computation of each nowcast (minutes)
timestep = 30
# the number of time steps for each nowcast (5 minutes for the MeteoSwiss and 
# FMI data)
num_timesteps = 24
# the threshold to use for precipitation/no precipitation
R_min = 0.1

if domain == "fmi":
    datasource = datasources.fmi
    precipevents = precipevents.fmi
else:
    datasource = datasources.mch
    precipevents = precipevents.mch

root_path = os.path.join(datasources.root_path, datasource["root_path"])
importer = io.get_method(datasource["importer"], "importer")

results = {}

for m in oflow_methods:
    results[m] = {}
    results[m]["CSI"] = [0.0]*num_timesteps
    results[m]["MAE"] = [0.0]*num_timesteps
    results[m]["n_samples"] = [0.0]*num_timesteps

R_min_dB = transformation.dB_transform(np.array([R_min]))[0][0]

for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")

    while curdate <= enddate:
        print("Computing nowcasts for event %d, start date %s..." % (pei+1, str(curdate)), end="")
        sys.stdout.flush()

        if curdate + num_timesteps * timedelta(minutes=5) > enddate:
            break

        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=9)

        R,_,metadata = io.readers.read_timeseries(fns, importer,
                                                  **datasource["importer_kwargs"])

        missing_data = False
        for i in range(R.shape[0]):
            if not np.any(np.isfinite(R[i, :, :])):
                print("Skipping, no finite values found for time step %d" % (i+1))
                missing_data = True
                break

        if missing_data:
            curdate += timedelta(minutes=timestep)
            continue

        R[~np.isfinite(R)] = metadata["zerovalue"]
        R = transformation.dB_transform(R, metadata=metadata)[0]

        obs_fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                          datasource["fn_pattern"], datasource["fn_ext"],
                                          datasource["timestep"], 
                                          num_next_files=num_timesteps)
        obs_fns = (obs_fns[0][1:], obs_fns[1][1:])
        if len(obs_fns[0]) == 0:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        R_obs,_,metadata = io.readers.read_timeseries(obs_fns, importer,
                                                      **datasource["importer_kwargs"])
        if R_obs is None:
            curdate += timedelta(minutes=timestep)
            print("Skipping, no verifying observations found.")
            continue

        for oflow_method in oflow_methods:
            oflow = motion.get_method(oflow_method)
            if oflow_method == "vet":
                R_ = R[-2:, :, :]
            else:
                R_ = R

            starttime = time.time()
            V = oflow(R_)
            print("%s optical flow computed in %.3f seconds." % \
                  (oflow_method, time.time() - starttime))

            if nowcast_method == "advection":
                nc = nowcasts.get_method("extrapolation")
                R_fct = nc(R[-1, :, :], V, num_timesteps)
            else:
                nc = nowcasts.get_method("steps")
                R_fct = nc(R[-3:, :, :], V, num_timesteps, noise_method=None, 
                           vel_pert_method=None, n_ens_members=1, 
                           mask_method="sprog", R_thr=R_min_dB, 
                           probmatching_method="mean", 
                           fft_method="numpy")[0, :, :, :]

            R_fct = transformation.dB_transform(R_fct, inverse=True)[0]

            for lt in range(num_timesteps):
                if not np.any(np.isfinite(R_obs[lt, :, :])):
                    print("Warning: no finite verifying observations for lead time %d." % (lt+1))
                    continue

                csi = det_cat_fcst(R_fct[lt, :, :], R_obs[lt, :, :], R_min, 
                                   ["CSI"])[0]
                MASK = np.logical_and(R_fct[lt, :, :] > R_min, 
                                      R_obs[lt, :, :] > R_min)
                if np.sum(MASK) == 0:
                    print("Skipping, no precipitation for lead time %d." % (lt+1))
                    continue

                mae = det_cont_fcst(R_fct[lt, :, :][MASK], R_obs[lt, :, :][MASK], 
                                     ["MAE_add"])[0]

                results[oflow_method]["CSI"][lt] += csi
                results[oflow_method]["MAE"][lt] += mae
                results[oflow_method]["n_samples"][lt] += 1

        print("Done.")

        # Save the intermediate results for testing purposes.
        with open("optflow_eval_results_%s.dat" % nowcast_method, "wb") as f:
            pickle.dump(results, f)

        curdate += timedelta(minutes=timestep)

with open("optflow_eval_results_%s.dat" % nowcast_method, "wb") as f:
    pickle.dump(results, f)
