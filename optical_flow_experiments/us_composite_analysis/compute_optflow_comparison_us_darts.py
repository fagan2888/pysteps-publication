"""
Runs the DARTS sensitivity experiments using parameters to compute the motion
field.

The data is stored in ./data/darts_tests folder.
"""

import sys
import time
from os.path import join

import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from skimage.transform import downscale_local_mean

from pysteps import io, motion, nowcasts
from pysteps.utils import transformation
from pysteps.verification.detcatscores import det_cat_fcst
from pysteps.verification.detcontscores import det_cont_fcst

# pyWAT is a WRF python toolkit currently being developed at McGill by
# Andres Perez Hortal.
# Since it is not stable enough, it is not made public yet.
from pywat.collections import LowAltCompositeCollection
from pywat.utils.helpers import get_timestamp
from pywat.utils.numerics import dbz_to_r

configurations = dict()
configurations[1] = dict()
configurations[2] = dict(N_x=50, Ny=50, M_x=3, M_y=3)
configurations[3] = dict(N_x=70, Ny=70, M_x=2, M_y=2)
configurations[4] = dict(N_x=70, Ny=50, M_x=2, M_y=2)
configurations[5] = dict(N_x=70, Ny=70, M_x=3, M_y=3)
configurations[6] = dict(N_x=100, Ny=100, M_x=3, M_y=2)
configurations[7] = dict(N_x=100, Ny=100, M_x=3, M_y=3)
configurations[8] = dict(N_x=50, Ny=50, M_x=4, M_y=4)


def create_dir(directory_path):
    """
    Create a directory if does not exists.
    """

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path)
        except Exception:

            raise Exception("Error creating directory: " + directory_path)


def get_lowalt_file(input_date):
    us_low_alt_dir = "/aos/home/aperez/localStorage/us_lowalt"

    file_name = input_date.strftime("us_lowalt_%Y%m%d_%H%M.nc.gz")

    destination_dir = os.path.join(us_low_alt_dir,
                                   input_date.strftime("%Y/%Y%m/%Y%m%d"))

    destination_path = os.path.join(destination_dir, file_name[:-3])

    return destination_path


def compute(nowcast_method, config_number):
    # the optical flow methods to use
    oflow_methods = ["darts"]

    # time step between computation of each nowcast (minutes)
    timestep = 30
    # the number of time steps for each nowcast (5 minutes for the MeteoSwiss and
    # FMI data)
    num_timesteps = 24
    # the threshold to use for precipitation/no precipitation
    R_min = 0.1

    R_min_dB = transformation.dB_transform(np.array([R_min]))[0][0]

    precip_events = [("201104160800", "201104170000"),
                     ("201111152300", "201111161000"),
                     ("201304110000", "201304120000"),
                     ("201304041800", "201304051800"),
                     ("201305180600", "201305181200"),
                     ("201305270000", "201305271200"),
                     ]

    for pei, pe in enumerate(precip_events):
        start_date = datetime.strptime(pe[0], "%Y%m%d%H%M")
        curdate = datetime.strptime(pe[0], "%Y%m%d%H%M")
        enddate = datetime.strptime(pe[1], "%Y%m%d%H%M")

        results = {}

        for m in oflow_methods:
            results[m] = {}
            results[m]["comptimes"] = 0.0
            results[m]["CSI"] = [0.0] * num_timesteps
            results[m]["RMSE"] = [0.0] * num_timesteps
            results[m]["n_samples"] = [0.0] * num_timesteps

        my_observations = LowAltCompositeCollection()

        while curdate <= enddate:
            print("Computing nowcasts for event %d, start date %s..." % (pei + 1, str(curdate)), end="")
            sys.stdout.flush()

            if curdate + num_timesteps * timedelta(minutes=5) > enddate:
                break

            time_step_in_sec = 5 * 60
            times = [curdate - timedelta(seconds=time_step_in_sec * i)
                     for i in range(9)]

            times += [curdate + timedelta(seconds=time_step_in_sec * i)
                      for i in range(1, num_timesteps + 1)]

            times.sort()

            # Add elements to the collection if they don't exists
            for _time in times:
                my_observations.add(get_lowalt_file(_time))

            # First 9 times
            R = my_observations.get_data('Reflectivity', date=times[:9])

            R = dbz_to_r(R, a=300., b=1.5)

            _R = list()

            # The original data is at 1km resolutions
            # Downscale to 5 km resolution by 5x5 averaging
            for i in range(9):
                _R.append(downscale_local_mean(R[i, :-1, :], (5, 5)))
            R = np.asarray(_R)
            my_observations.clean_buffers()  # release memory

            missing_data = False
            for i in range(R.shape[0]):
                if not np.any(np.isfinite(R[i, :, :])):
                    print("Skipping, no finite values found for time step %d" % (i + 1))
                    missing_data = True
                    break

            if missing_data:
                curdate += timedelta(minutes=timestep)
                continue

            R = transformation.dB_transform(R)[0]

            # Forecast times
            fcts_times = times[9:]
            R_obs = my_observations.get_data('Reflectivity', date=times[9:])
            R_obs = dbz_to_r(R_obs, a=300., b=1.5)

            # The original data is at 1km resolutions
            # Downscale to 5 km resolution by 5x5 averaging
            _R = list()
            for i in range(len(fcts_times)):
                _R.append(downscale_local_mean(R_obs[i, :-1, :], (5, 5)))
            R_obs = np.asarray(_R)
            my_observations.clean_buffers()  # release memory

            for oflow_method in oflow_methods:
                oflow = motion.get_method(oflow_method)
                if oflow_method == "vet":
                    R_ = R[-2:, :, :]
                else:
                    R_ = R

                starttime = time.time()
                V = oflow(R_, **configurations[config_number])
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
                        print("Warning: no finite verifying observations for lead time %d." % (lt + 1))
                        continue

                    csi = det_cat_fcst(R_fct[lt, :, :], R_obs[lt, :, :], R_min,
                                       ["CSI"])[0]
                    MASK = np.logical_and(R_fct[lt, :, :] > R_min,
                                          R_obs[lt, :, :] > R_min)
                    if np.sum(MASK) == 0:
                        print("Skipping, no precipitation for lead time %d." % (lt + 1))
                        continue

                    rmse = det_cont_fcst(R_fct[lt, :, :][MASK], R_obs[lt, :, :][MASK],
                                         ["MAE_add"])[0]

                    results[oflow_method]["CSI"][lt] += csi
                    results[oflow_method]["RMSE"][lt] += rmse
                    results[oflow_method]["n_samples"][lt] += 1

            print("Done.")

            curdate += timedelta(minutes=timestep)

        data_dir = './data/dart_tests/config_{:d}'.format(config_number)
        create_dir(data_dir)
        file_name = "optflow_comparison_results_%s_%s.dat" % (nowcast_method,
                                                              get_timestamp(start_date))
        with open(join(data_dir, file_name), "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":

    for config_number in configurations:
        print(config_number)
        compute("advection", config_number)
        compute("sprog", config_number)
