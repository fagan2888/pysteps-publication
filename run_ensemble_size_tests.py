# Runs STEPS with different ensemble sizes and computes probabilistic 
# verification statistics with different intensity thresholds (Figures 9,10,16 
# and 17 in the paper).

from collections import defaultdict
from datetime import datetime, timedelta
import os
import pickle
import sys
import time
import dask
import numpy as np
from scipy.ndimage.filters import uniform_filter
from pysteps import io, motion, nowcasts, utils
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, transformation
from pysteps.verification import ensscores
from pysteps.verification import probscores
import datasources, precipevents

ensemble_sizes = [96, 48, 24, 12, 6]
#domain = "fmi"
domain = "mch"
timestep = 30
num_workers = 6
num_timesteps = 36
R_min = 0.1
R_thrs = [0.1, 1.0, 5.0, 10.0]
upscale_factor = 1

vp_par  = (2.31970635, 0.33734287, -2.64972861)
vp_perp = (1.90769947, 0.33446594, -2.06603662)

if domain == "fmi":
    datasource = datasources.fmi
    precipevents = precipevents.fmi
else:
    datasource = datasources.mch
    precipevents = precipevents.mch

root_path = os.path.join(datasources.root_path, datasource["root_path"])
importer = io.get_method(datasource["importer"], "importer")

results = {}

def upscale_precip_field(R, sf):
    if sf % 2 == 0:
        origin = -1
    else:
        origin = 0

    R = R.copy()
    MASK = np.isfinite(R)
    R[~MASK] = 0.0

    R1 = uniform_filter(R, sf, origin=origin, mode="nearest")[::sf, ::sf]
    R2 = uniform_filter(MASK.astype(float), sf, origin=origin, mode="nearest")[::sf, ::sf]

    R_u =  R1 / R2
    R_u[R2 < 0.5] = np.nan

    return R_u

for es in ensemble_sizes:
    results[es] = {}

    results[es]["cat"] = {}
    results[es]["MAE"] = {}
    results[es]["ME"] = {}

    results[es]["CRPS"] = {}
    results[es]["rankhist"] = {}
    results[es]["reldiag"] = {}
    results[es]["ROC"] = {}

    for lt in range(num_timesteps):
        results[es]["CRPS"][lt] = probscores.CRPS_init()

    for R_thr in R_thrs:
        results[es]["cat"][R_thr] = {}
        results[es]["MAE"][R_thr] = {}
        results[es]["ME"][R_thr] = {}

        results[es]["rankhist"][R_thr] = {}
        results[es]["reldiag"][R_thr] = {}
        results[es]["ROC"][R_thr] = {}

        for lt in range(num_timesteps):
            results[es]["cat"][R_thr][lt] = {"H":0, "F":0, "M":0, "R":0}
            results[es]["MAE"][R_thr][lt] = {"sum": 0.0, "n":0}
            results[es]["ME"][R_thr][lt] = {"sum": 0.0, "n":0}

            results[es]["rankhist"][R_thr][lt] = ensscores.rankhist_init(es, R_thr)
            results[es]["reldiag"][R_thr][lt] = probscores.reldiag_init(R_thr, n_bins=10)
            results[es]["ROC"][R_thr][lt] = probscores.ROC_curve_init(R_thr, n_prob_thrs=100)

R_min_dB = transformation.dB_transform(np.array([R_min]))[0][0]

outfn = "ensemble_size_results_%s" % domain
if upscale_factor > 1:
    outfn += "_%d" % upscale_factor

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
        if domain == "fmi":
            R, metadata = conversion.to_rainrate(R, metadata, a=223.0, b=1.53)
        if upscale_factor > 1:
            R_ = []
            for i in range(R.shape[0]):
                R_.append(upscale_precip_field(R[i, :, :], upscale_factor))
            R = np.stack(R_)

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

        if domain == "fmi":
            R_obs, metadata = conversion.to_rainrate(R_obs, metadata, a=223.0, b=1.53)

        if upscale_factor > 1:
            R_ = []
            for i in range(R_obs.shape[0]):
                R_.append(upscale_precip_field(R_obs[i, :, :], upscale_factor))
            R_obs = np.stack(R_)

        for es in ensemble_sizes:
            oflow = motion.get_method("lucaskanade")
            V = oflow(R[-2:, :, :])

            nc = nowcasts.get_method("steps")
            vel_pert_kwargs = {"p_par":vp_par , "p_perp":vp_perp}
            R_fct = nc(R[-3:, :, :], V, num_timesteps, n_ens_members=es,
                       n_cascade_levels=8, R_thr=R_min_dB,
                       kmperpixel=1.0*upscale_factor, timestep=5,
                       vel_pert_method="bps", mask_method="incremental",
                       num_workers=num_workers, fft_method="pyfftw",
                       vel_pert_kwargs=vel_pert_kwargs)

            for ei in range(R_fct.shape[0]):
                for lt in range(R_fct.shape[1]):
                    R_fct[ei, lt, :, :] = \
                        transformation.dB_transform(R_fct[ei, lt, :, :], inverse=True)[0]

            R_fct_mean = np.mean(R_fct, axis=0)

            def worker1(lt):
                probscores.CRPS_accum(results[es]["CRPS"][lt], R_fct[:, lt, :, :],
                                      R_obs[lt, :, :])

            def worker2(lt, R_thr):
                if not np.any(np.isfinite(R_obs[lt, :, :])):
                    return

                P_fct = excprob(R_fct[:, lt, :, :], R_thr)

                ensscores.rankhist_accum(results[es]["rankhist"][R_thr][lt],
                                         R_fct[:, lt, :, :], R_obs[lt, :, :])
                probscores.reldiag_accum(results[es]["reldiag"][R_thr][lt],
                                         P_fct, R_obs[lt, :, :])
                probscores.ROC_curve_accum(results[es]["ROC"][R_thr][lt],
                                           P_fct, R_obs[lt, :, :])

            def worker3(lt, R_thr):
                MASK_f = R_fct_mean[lt, :, :] > R_thr
                MASK_o = R_obs[lt, :, :] > R_thr

                H = np.logical_and(MASK_f, MASK_o)
                F = np.logical_and(MASK_f, ~MASK_o)
                M = np.logical_and(~MASK_f, MASK_o)
                R = np.logical_and(~MASK_f, ~MASK_o)

                MASK = np.logical_and(MASK_f, MASK_o)
                n = np.sum(MASK)
                me = R_fct_mean[lt, :, :][MASK] - R_obs[lt, :, :][MASK]

                results[es]["cat"][R_thr][lt]["H"] += np.sum(H)
                results[es]["cat"][R_thr][lt]["F"] += np.sum(F)
                results[es]["cat"][R_thr][lt]["M"] += np.sum(M)
                results[es]["cat"][R_thr][lt]["R"] += np.sum(R)
                results[es]["ME"][R_thr][lt]["sum"] += np.sum(me)
                results[es]["ME"][R_thr][lt]["n"] += n
                results[es]["MAE"][R_thr][lt]["sum"] += np.sum(np.abs(me))
                results[es]["MAE"][R_thr][lt]["n"] += n

            res = []
            for i,R_thr in enumerate(R_thrs):
                for lt in range(num_timesteps):
                    if not np.any(np.isfinite(R_obs[lt, :, :])):
                        print("Warning: no finite verifying observations for lead time %d." % (lt+1))
                        continue
                    if i == 0:
                        res.append(dask.delayed(worker1)(lt))
                    res.append(dask.delayed(worker2)(lt, R_thr))
                    res.append(dask.delayed(worker3)(lt, R_thr))
            dask.compute(*res, num_workers=num_workers)

        with open(outfn + ".dat", "wb") as f:
            pickle.dump(results, f)

        curdate += timedelta(minutes=timestep)

with open(outfn + ".dat", "wb") as f:
    pickle.dump(results, f)
