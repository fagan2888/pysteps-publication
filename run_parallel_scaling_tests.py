# Runs STEPS with different ensemble sizes and measures computation times.
# (excluding the time for computing the optical flow and saving the output to
# disk).

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import sys
import time
from pysteps import io, motion, nowcasts
from pysteps.utils import transformation
import datasources
import precipevents

# the ensemble sizes to use
ensemble_sizes = [6, 12, 24, 48]
# maximum number of threads to use
max_num_threads = 12
# the domain: "fmi" or "mch"
domain = "fmi"
# number of nowcast time steps to compute
num_timesteps = 12
# threshold for precipitation/no precipitation
R_min = 0.1
# the FFT method to use
fft_method = "pyfftw"

if domain == "fmi":
    datasource = datasources.fmi
    precipevents = precipevents.fmi
else:
    datasource = datasources.mch
    precipevents = precipevents.mch

root_path = os.path.join(datasources.root_path, datasource["root_path"])
importer = io.get_method(datasource["importer"], "importer")

results = defaultdict(dict)

R_min_dB = transformation.dB_transform(np.array([R_min]))[0][0]

for pei, pe in enumerate(precipevents):
    curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
    enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")

    while curdate <= enddate:
        print("Computing nowcasts for event %d, start date %s..." % (pei+1, str(curdate)), end="")
        sys.stdout.flush()

        if curdate + num_timesteps * timedelta(minutes=5) > enddate:
            break

        fns = io.archive.find_by_date(curdate, root_path, datasource["path_fmt"],
                                      datasource["fn_pattern"], datasource["fn_ext"],
                                      datasource["timestep"], num_prev_files=2)

        R, _, metadata = io.readers.read_timeseries(fns, importer,
                                                    **datasource["importer_kwargs"])

        missing_data = False
        for i in range(R.shape[0]):
            if not np.any(np.isfinite(R[i, :, :])):
                print("Skipping, no finite values found for time step %d" % (i+1))
                missing_data = True
                break

        if missing_data:
            curdate += timedelta(minutes=datasource["timestep"])
            continue

        R[~np.isfinite(R)] = metadata["zerovalue"]
        R = transformation.dB_transform(R, metadata=metadata)[0]

        oflow = motion.get_method("lucaskanade")
        V = oflow(R[-2:, :, :])

        nc = nowcasts.get_method("steps")

        for es in ensemble_sizes:
            for nw in range(1, max_num_threads+1, 1):
                starttime = time.time()
                _, init_time, mainloop_time = \
                  nc(R[-3:, :, :], V, num_timesteps, n_ens_members=es,
                  n_cascade_levels=6, R_thr=R_min_dB, kmperpixel=1.0,
                  timestep=datasource["timestep"], vel_pert_method="bps",
                  mask_method="incremental", probmatching_method="cdf",
                  num_workers=nw, fft_method=fft_method, measure_time=True)
                results[es][nw] = (init_time, mainloop_time)

                # Save the intermediate results for testing purposes.
                with open("parallel_scaling_results_%s.dat" % domain, "wb") as f:
                    pickle.dump(dict(results), f)

        #curdate += timedelta(minutes=datasource["timestep"])
        # This script terminates after the first event. No averaging over
        # different events implemented yet.
        exit()

with open("parallel_scaling_results_%s.dat" % domain, "wb") as f:
    pickle.dump(dict(results), f)
