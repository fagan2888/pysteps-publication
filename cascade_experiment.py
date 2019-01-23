#!/bin/env python

"""Generation and verification of an ensemble nowcast

The script shows how to run verification experiments for ensemble precipitation
nowcasting with pysteps.

More info: https://pysteps.github.io/
"""
import csv
import datetime
import matplotlib.pylab as plt
import netCDF4
import numpy as np
import os
import pprint
import sys
import time

import matplotlib
matplotlib.use('Agg')

import pysteps as stp

# Verification settings
verification = {
    "experiment_name"   : "new_cascade_mask_bps",
    "overwrite"         : False,            # to recompute nowcasts
    "v_thresholds"      : [.1, 2., 5.],    # [mm/h]
    "v_leadtimes"       : [60,120],         # [min]
    "v_accu"            : 5,               # [min]
    "seed"              : 42,               # for reproducibility
    "doplot"            : True,             # save figures
    "dosaveresults"     : True              # save verification scores to csv
}

# Forecast settings
forecast = {
    "min_war"           : 0.05,
    "min_imf"           : 0.05,
    "n_lead_times"      : 24,       # timesteps per nowcast
    "r_threshold"       : 0.1,      # rain/no rain threshold [mm/h]
    "unit"              : "mm/h",
    "transformation"    : "dB",
    "zerovalue"         : -15
}

# Events   event start     event end       update cycle  data source
data_source = "mch_aqc"
time_step_min = 120
events = [("201604161800", "201604170600", time_step_min,       data_source)]
# events = [("201604161800", "201604170600", time_step_min,       data_source),
          # ("201607111300", "201607120100", time_step_min,       data_source),
          # ("201701311000", "201701312200", time_step_min,       data_source),
          # ("201706141300", "201706150100", time_step_min,       data_source),
          # ("201706242200", "201706251000", time_step_min,       data_source),
          # ("201706272000", "201706280800", time_step_min,       data_source),
          # ("201707191300", "201707200100", time_step_min,       data_source),
          # ("201707211300", "201707220100", time_step_min,       data_source),
          # ("201707291300", "201707300100", time_step_min,       data_source),
          # ("201708311400", "201709010200", time_step_min,       data_source)]
          
# read data quality
badts2016 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2016_radar-stats-badTimestamps_00005.txt", dtype="str")
badts2017 = np.loadtxt("/store/msrad/radar/precip_attractor/radar_availability/AQC-2017_radar-stats-badTimestamps_00005.txt", dtype="str")
badts = np.concatenate((badts2016, badts2017))

# The experiment set-up
## this includes tuneable parameters

if "bps" in verification["experiment_name"]:
    motion_pert = "bps"
else:
    motion_pert = None

experiment = {

    ## the methods
    "oflow_method"      : ["lucaskanade"],      # lucaskanade, darts
    "adv_method"        : ["semilagrangian"],   # semilagrangian, eulerian
    "nwc_method"        : ["steps"],
    "noise_method"      : ["nonparametric"],    # parametric, nonparametric, ssft
    "decomp_method"     : ["fft"],

    ## the parameters
    "n_ens_members"     : [24],
    "ar_order"          : [2],
    "n_cascade_levels"  : [1,8],
    "mask_method"       : ["incremental", None], # obs, incremental, sprog, None
    "prob_matching"     : ["cdf"],
    "shift_and_scale"   : [False],
    "vel_pert_method"   : [motion_pert]
}

# Set the BPS motion perturbation parameters that are adapted to the Swiss domain
if motion_pert == "bps":
    print("Using Swiss parameters for motion perturbation.")
    vel_pert_kwargs = {"p_pert_par":(2.56,0.33,-3.0), "p_pert_perp":(1.31,0.36,-1.02)}
else:
    print("Using default parameters for motion perturbation.")
    vel_pert_kwargs = {} # Will use the default parameters

# Conditional parameters
## parameters that can be directly related to other parameters
def cond_pars(pars):
    for key in list(pars):
        if key == "oflow_method":
            if pars[key].lower() == "darts":  pars["n_prvs_times"] = 9
            else:                             pars["n_prvs_times"] = 3
        elif key.lower() == "n_cascade_levels":
            if pars[key] == 1 : pars["bandpass_filter"] = "uniform"
            else:               pars["bandpass_filter"] = "gaussian"
        elif key.lower() == "nwc_method":
            if pars[key] == "extrapolation" : pars["n_ens_members"] = 1
    return pars

# Prepare the list of all parameter sets of the verification
parsets = [[]]
for _, items in experiment.items():
    parsets = [parset+[item] for parset in parsets for item in items]

# Now loop all parameter sets
for n, parset in enumerate(parsets):
    
    tic = time.time()

    # Build parameter set
    p = {}
    for m, key in enumerate(experiment.keys()):
        p[key] = parset[m]
    ## apply conditional parameters
    p = cond_pars(p)
    ## include all remaining parameters
    p.update(verification)
    p.update(forecast)

    print("************************")
    print("* Parameter set %02d/%02d: *" % (n+1, len(parsets)))
    print("************************")

    pprint.pprint(p)

    # If necessary, build path to results
    path_to_experiment = os.path.join(stp.rcparams.outputs["path_outputs"], p["experiment_name"])
    # subdir with event date
    path_to_nwc = path_to_experiment
    for key, item in p.items():
        # include only variables that change
        if len(experiment.get(key,[None])) > 1:
            path_to_nwc = os.path.join(path_to_nwc, '-'.join([key, str(item)]))

    print(path_to_nwc)

    try:
        os.makedirs(path_to_nwc)
    except FileExistsError:
        pass

    # **************************************************************************
    # NOWCASTING
    # **************************************************************************

    # Loop events
    countnwc = 0
    for event in events:

        ## import data specifications
        ds = stp.rcparams.data_sources[event[3]]

        if p["v_accu"] is None:
            p["v_accu"] = ds.timestep

        # Loop forecasts within given event using the prescribed update cycle interval
        startdate   = datetime.datetime.strptime(event[0], "%Y%m%d%H%M")
        enddate     = datetime.datetime.strptime(event[1], "%Y%m%d%H%M")
        
        while startdate + datetime.timedelta(minutes = p["n_lead_times"]*ds.timestep) < enddate:

            # filename of the nowcast netcdf
            outfn = os.path.join(path_to_nwc, "%s_%s_nowcast.netcdf" % (startdate.strftime("%Y%m%d%H%M"),event[3]))

            ## check if results already exists
            run_exist = False
            if os.path.isfile(outfn):
                fid = netCDF4.Dataset(outfn, 'r')
                if fid.dimensions["time"].size == p["n_lead_times"]:
                    run_exist = True
                    if p["overwrite"]:
                        os.remove(outfn)
                        run_exist = False
                else:
                    os.remove(outfn)

            if run_exist:
                print("Nowcast %s_nowcast already exists in %s" % (startdate.strftime("%Y%m%d%H%M"),path_to_nwc))

            else:

                print("Starttime: %s" % startdate.strftime("%Y%m%d%H%M"))

                ## redirect stdout to log file
                logfn =  os.path.join(path_to_nwc, "%s_%s_log.txt" % (startdate.strftime("%Y%m%d%H%M"),event[3]))
                print("Log: %s" % logfn)
                orig_stdout = sys.stdout
                f = open(logfn, 'w')
                sys.stdout = f

                print("*******************")
                print("* %s *****" % startdate.strftime("%Y%m%d%H%M"))
                print("* Parameter set : *")
                pprint.pprint(p)
                print("*******************")

                print("--- Start of the run : %s ---" % (datetime.datetime.now()))

                ## time
                t0 = time.time()

                # Read inputs
                print("Read the data...")

                ## find radar field filenames
                input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                                  ds.fn_ext, ds.timestep, p["n_prvs_times"])


                ## read radar field files
                importer    = stp.io.get_method(ds.importer, "importer")
                R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
                metadata0 = metadata.copy()
                metadata0["shape"] = R.shape[1:]

                # Check observation quality
                badobs = []
                for tt in metadata["timestamps"]:
                    badobs.append(tt.strftime("%Y%m%d%H%M") in badts)
                badobs = np.any(np.array(badobs))
                print("Bad obs = %s" % badobs)

                # Prepare input files
                print("Prepare the data...")

                ## if necessary, convert to rain rates [mm/h]
                converter = stp.utils.get_method("mm/h")
                R, metadata = converter(R, metadata)

                ## threshold the data
                R[R < p["r_threshold"]] = 0.0
                metadata["threshold"] = p["r_threshold"]
                
                # Check imf
                R_ = R[-1, :, :]
                this_imf = np.nanmean(R_[R_ > 0])

                # Check war
                this_war = (R[-1, :, :] > metadata["zerovalue"]).sum() / np.isfinite(R[-1, :, :]).sum()
                print("WAR = %.3f" % this_war)

                if this_war > p["min_war"] and this_imf > p["min_imf"] and not badobs:
                    
                    countnwc += 1
                    
                    ## transform the data
                    transformer = stp.utils.get_method(p["transformation"])
                    R, metadata = transformer(R, metadata, zerovalue=p["zerovalue"])

                    # Perform the nowcast

                    ## set NaN equal to zero
                    R[~np.isfinite(R)] = metadata["zerovalue"]

                    # Compute motion field
                    oflow_method = stp.motion.get_method(p["oflow_method"])
                    UV = oflow_method(R)

                    ## define the callback function to export the nowcast to netcdf
                    converter = stp.utils.get_method("mm/h")
                    def export(X):
                        X,_ = converter(X, metadata)
                        stp.io.export_forecast_dataset(X, exporter)
                    
                    # Update metadata so that it is consistent!
                    metadata0["unit"] = 'mm/h'

                    ## initialize netcdf file
                    incremental = "timestep"
                    exporter = stp.io.initialize_forecast_exporter_netcdf(outfn, startdate,
                                      ds.timestep, p["n_lead_times"], metadata0["shape"],
                                      p["n_ens_members"], metadata0, incremental=incremental)

                    #%% start the nowcast
                    print("Computing the nowcast (%02d) ..." % countnwc)
                    nwc = stp.nowcasts.get_method(p["nwc_method"])
                    nwc(R, UV, p["n_lead_times"],
                            p["n_ens_members"], p["n_cascade_levels"],
                            kmperpixel=metadata["xpixelsize"]/1000,
                            timestep=ds.timestep,  R_thr=metadata["threshold"],
                            extrap_method=p["adv_method"],
                            decomp_method=p["decomp_method"],
                            bandpass_filter_method=p["bandpass_filter"],
                            noise_method=p["noise_method"],
                            ar_order=p["ar_order"],
                            probmatching_method=p["prob_matching"],
                            mask_method=p["mask_method"],
                            vel_pert_method=p["vel_pert_method"],
                            callback=export, return_output=False, seed=p["seed"], vel_pert_kwargs=vel_pert_kwargs)                   

                    ## save results
                    stp.io.close_forecast_file(exporter)

                else:
                    print("Run aborted.")

                # save log
                print("--- End of the run : %s ---" % (datetime.datetime.now()))
                print("--- Total time : %s seconds ---" % (time.time() - t0))
                sys.stdout = orig_stdout
                f.close()

            # next forecast
            startdate += datetime.timedelta(minutes = event[2])

    # **************************************************************************
    # VERIFICATION
    # **************************************************************************

    rankhists = {}
    reldiags = {}
    rocs = {}
    for lt in p["v_leadtimes"]:
        for thr in p["v_thresholds"]:
            rankhists[lt, thr] = stp.verification.ensscores.rankhist_init(p["n_ens_members"], thr)    
            reldiags[lt, thr]  = stp.verification.probscores.reldiag_init(thr)
            rocs[lt, thr] = stp.verification.probscores.ROC_curve_init(thr)

    # Loop the events
    countnwc = 0
    for event in events:

        # Loop the forecasts
        startdate   = datetime.datetime.strptime(event[0], "%Y%m%d%H%M")
        enddate     = datetime.datetime.strptime(event[1], "%Y%m%d%H%M")
        
        while startdate + datetime.timedelta(minutes = p["n_lead_times"]*ds.timestep) < enddate:
            countnwc+=1

            print("Verifying the nowcast (%02d) ..." % countnwc)

            # Read observations

            ## find radar field filenames
            input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                              ds.fn_ext, ds.timestep, 0, p["n_lead_times"])

            ## read radar field files
            importer = stp.io.get_method(ds.importer, "importer")
            R_obs, _, metadata_obs = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
            R_obs = R_obs[1:,:,:]
            metadata_obs["timestamps"] = metadata_obs["timestamps"][1:]

            # check observation quality
            badobs = []
            for tt in metadata_obs["timestamps"]:
                badobs.append(tt.strftime("%Y%m%d%H%M") in badts)
            badobs = np.array(badobs)
            badops_propr = badobs.sum()/badobs.size

            if badops_propr < .1:
            
                # remove bad observations
                R_obs[badobs, :, :] *= 0.0
                
                ## if necessary, convert to rain rates [mm/h]
                converter = stp.utils.get_method("mm/h")
                R_obs, metadata_obs = converter(R_obs, metadata_obs)

                ## threshold the data
                R_obs[R_obs < p["r_threshold"]] = 0.0
                metadata_obs["threshold"] = p["r_threshold"]

                # Load the nowcast

                ## filename of the nowcast netcdf
                infn = os.path.join(path_to_nwc, "%s_%s_nowcast.netcdf" % (startdate.strftime("%Y%m%d%H%M"),event[3]))

                print("     read: %s" % infn)
                if os.path.isfile(infn):
                    ## read netcdf
                    R_fct, metadata_fct = stp.io.import_netcdf_pysteps(infn)
                    timestamps = metadata_fct["timestamps"]
                    leadtimes = np.arange(1,len(timestamps)+1)*ds.timestep # min
                    metadata_fct["leadtimes"] = leadtimes
                    
                    # Correct wrongly written metadata
                    metadata_fct["unit"] = "mm/h"
                    
                    ## threshold the data
                    R_fct[R_fct < p["r_threshold"]] = 0.0
                    metadata_fct["threshold"] = p["r_threshold"]
                    
                    # remove timestamps with bad obs
                    R_fct[:, badobs, :, :] *= 0.0

                    ## if needed, compute accumulations
                    aggregator = stp.utils.get_method("accumulate")
                    R_obs, metadata_obs = aggregator(R_obs, metadata_obs, p["v_accu"])
                    R_fct, metadata_fct = aggregator(R_fct, metadata_fct, p["v_accu"])
                    leadtimes = metadata_fct["leadtimes"]
                    
                    ## Re-threshold the accumulated data. Important!
                    R_fct[R_fct < p["r_threshold"]] = 0.0
                    metadata_fct["threshold"] = p["r_threshold"]
                    
                    R_obs[R_obs < p["r_threshold"]] = 0.0
                    metadata_obs["threshold"] = p["r_threshold"]
                    
                    #%% Do verification

                    ## loop leadtimes
                    for i,lt in enumerate(p["v_leadtimes"]):

                        idlt = leadtimes == lt
                        
                        ## loop thresholds
                        for thr in p["v_thresholds"]:
                            ## rank histogram
                            stp.verification.ensscores.rankhist_accum(rankhists[lt,thr], R_fct[:, idlt, :, :], R_obs[idlt, :, :])
                            
                            ## Compute fx probabilities
                            P_fct = stp.postprocessing.ensemblestats.excprob(R_fct[:, idlt, :, :], thr, ignore_nan=True)
                            
                            ## reliability diagram
                            stp.verification.probscores.reldiag_accum(reldiags[lt, thr], P_fct, R_obs[idlt, :, :])
                            ## roc curve
                            stp.verification.probscores.ROC_curve_accum(rocs[lt, thr], P_fct, R_obs[idlt, :, :])
                else:
                    print("     not found.")
            else:
                print("     low quality observations.")

            ## next forecast
            startdate += datetime.timedelta(minutes = event[2])

    # Write out and plot verification scores for parameter set
    for i,lt in enumerate(p["v_leadtimes"]):

        idlt = leadtimes == lt
        
        for thr in p["v_thresholds"]:

            if verification["dosaveresults"]:
                
                ## write rank hist results to csv file
                fn = os.path.join(path_to_nwc, "rankhist_%03d_%03d_thr%.1f.csv" % (lt, p["v_accu"], thr))
                with open(fn, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in rankhists[lt,thr].items():
                       writer.writerow([key, value])
                   
                ## write rel diag results to csv file
                fn = os.path.join(path_to_nwc, "reldiag_%03d_%03d_thr%.1f.csv" % (lt, p["v_accu"], thr))
                with open(fn, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in reldiags[lt, thr].items():
                       writer.writerow([key, value])

                ## write roc curve results to csv file
                fn = os.path.join(path_to_nwc, "roc_%03d_%03d_thr%.1f.csv" % (lt, p["v_accu"], thr))
                with open(fn, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in rocs[lt, thr].items():
                       writer.writerow([key, value])

            if verification["doplot"]:
                fig = plt.figure()
                stp.verification.plot_rankhist(rankhists[lt,thr], ax=fig.gca())
                plt.tight_layout()
                plt.savefig(os.path.join(path_to_nwc, "rankhist_%03d_%03d_thr%.1f.png" % (lt, p["v_accu"], thr)))
                plt.close()
            
                fig = plt.figure()
                stp.verification.plot_reldiag(reldiags[lt, thr], ax=fig.gca())
                plt.tight_layout()
                plt.savefig(os.path.join(path_to_nwc, "reldiag_%03d_%03d_thr%.1f.png" % (lt, p["v_accu"], thr)))
                plt.close()

                fig = plt.figure()
                stp.verification.plot_ROC(rocs[lt, thr], ax=fig.gca())
                plt.tight_layout()
                plt.savefig(os.path.join(path_to_nwc, "roc_%03d_%03d_thr%.1f.png" % (lt, p["v_accu"], thr)))
                plt.close()
                
    toc = time.time()
    print('Elapsed time for one parameter setting:', toc-tic)
print('Finished!')