
from collections import defaultdict
from datetime import datetime, timedelta
import os
import pickle
import sys
import numpy as np
import datasources, precipevents
from pysteps import cascade, extrapolation, io, motion
from pysteps.timeseries import autoregression

# the domain: "fmi" or "mch"
domain = "fmi"
# the optical flow method to use
oflow_method = "lucaskanade"
# the number of nowcast timesteps to use in the analysis
num_timesteps = 36
# time step between computation of each nowcast (minutes)
timestep = 30
# the number of cascade levels to use
num_cascade_levels = 8
# threshold value for precipitation/no precipitation and value for no
# precipitation
# TODO: These values are chose assuming that the input data is in dBZ, so
# without applying conversion this script currently works only for the FMI data.
R_thr = (10.0, 5.0)

if domain == "fmi":
    datasource = datasources.fmi
    precipevents = precipevents.fmi
else:
    datasource = datasources.mch
    precipevents = precipevents.mch

root_path = os.path.join(datasources.root_path, datasource["root_path"])
importer = io.get_method(datasource["importer"], "importer")

results = {}
results["leadtimes"] = (np.arange(num_timesteps)+1) * datasource["timestep"]
results["timestep"] = datasource["timestep"]

results["cc_ar"] = [np.zeros(num_timesteps) for i in range(num_cascade_levels)]
results["cc_obs"] = [np.zeros(num_timesteps) for i in range(num_cascade_levels)]
results["n_ar_samples"] = [np.zeros(num_timesteps, dtype=int) \
                           for i in range(num_cascade_levels)]
results["n_obs_samples"] = [np.zeros(num_timesteps, dtype=int) \
                            for i in range(num_cascade_levels)]

filter = None
oflow = motion.get_method(oflow_method)
extrap_init = extrapolation.get_method("semilagrangian")[0]
extrapolator = None

for pei,pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")
    
    while curdate <= enddate:
        print("Running analyses for event %d, start date %s..." % (pei+1, str(curdate)), end="")
        sys.stdout.flush()

        if curdate + num_timesteps * timedelta(minutes=5) > enddate:
            print("Done.")
            break
        
        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=2)

        R,_,metadata = io.readers.read_timeseries(fns, importer,
                                                  **datasource["importer_kwargs"])

        # TODO: Convert the data to mm/h.

        missing_data = False
        for i in range(R.shape[0]):
            if not np.any(np.isfinite(R[i, :, :])):
                print("Skipping, no finite values found for time step %d" % (i+1))
                missing_data = True
                break

        R[~np.isfinite(R)] = R_thr[1]
        R[R < R_thr[0]] = R_thr[1]
        
        if filter is None:
          filter = cascade.bandpass_filters.filter_gaussian(R.shape[1:], num_cascade_levels)

        if extrapolator is None:
            extrapolator = extrap_init(shape=R.shape[1:])

        # TODO: Supply enough input fields when DARTS is used.
        V = oflow(R[-2:, :, :])

        R_minus_2 = extrapolation.semilagrangian.extrapolate(extrapolator, R[-3, :, :], V, 2,
                                                             outval=R_thr[1])[-1, :, :]
        R_minus_1 = extrapolation.semilagrangian.extrapolate(extrapolator, R[-2, :, :], V, 1,
                                                             outval=R_thr[1])[-1, :, :]

        c1 = cascade.decomposition.decomposition_fft(R_minus_2, filter)
        c2 = cascade.decomposition.decomposition_fft(R_minus_1, filter)
        c3 = cascade.decomposition.decomposition_fft(R[-1, :, :], filter)

        gamma = []
        phi = []
        for i in range(num_cascade_levels):
            gamma_1 = np.corrcoef(c3["cascade_levels"][i, :, :].flatten(),
                                  c2["cascade_levels"][i, :, :].flatten())[0, 1]
            gamma_2 = np.corrcoef(c3["cascade_levels"][i, :, :].flatten(),
                                  c1["cascade_levels"][i, :, :].flatten())[0, 1]
            gamma_2 = autoregression.adjust_lag2_corrcoef2(gamma_1, gamma_2)
            gamma.append((gamma_1, gamma_2))

            phi.append(autoregression.estimate_ar_params_yw((gamma_1, gamma_2))[0:2])

        for i in range(num_cascade_levels):
            k = 0
            for t in range(num_timesteps):
                # TODO: Implement this for a higher-order AR(p) model.
                if k == 0:
                    rho = [np.nan, gamma[i][0]]
                elif k == 1:
                    rho = [gamma[i][0], gamma[i][1]]
                else:
                    rho.append(rho[1] * phi[i][0] + rho[0] * phi[i][1])
                    rho.pop(0)
                if np.isfinite(rho[1]):
                  results["cc_ar"][i][t] += rho[1]
                  results["n_ar_samples"][i][t] += 1
                k += 1

        R_ep = extrapolation.semilagrangian.extrapolate(extrapolator, R[-1, :, :], 
            V, num_timesteps, outval=R_thr[1])

        obs_fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                          datasource["fn_pattern"], datasource["fn_ext"],
                                          datasource["timestep"], num_next_files=num_timesteps)
        R_obs,_,_ = io.read_timeseries(obs_fns, importer, **datasource["importer_kwargs"])
        R_obs = R_obs[1:, :, :]
        MASK_obs = np.isfinite(R_obs[-1, :, :])
        R_obs[~np.isfinite(R_obs)] = R_thr[1]
        R_obs[R_obs < R_thr[0]] = R_thr[1]

        for t in range(num_timesteps):
            c_f = cascade.decomposition.decomposition_fft(R_ep[t, :, :], filter)
            R_ep[t, ~MASK_obs] = R_thr[1]
            c_o = cascade.decomposition.decomposition_fft(R_obs[t, :, :], filter)
            
            for i in range(num_cascade_levels):
                cc = np.corrcoef(c_f["cascade_levels"][i, :, :].flatten(),
                                 c_o["cascade_levels"][i, :, :].flatten())[0, 1]
                if np.isfinite(cc):
                    results["cc_obs"][i][t] += cc
                    results["n_obs_samples"][i][t] += 1

        print("Done.")

        curdate += timedelta(minutes=timestep)

        with open("ar2_corr_results_%s.dat" % domain, "wb") as f:
            pickle.dump(results, f)

with open("ar2_corr_results_%s.dat" % domain, "wb") as f:
    pickle.dump(results, f)
