#!/bin/bash

# Example bash script that starts a job with srun to compute pysteps experiments.
# To enable dask parallelization OMP_NUM_THREADS=1 and -c 24 (=number of workers in pysteps)

python_script=run_nowcast_verification.py
expname=cascade_exp_accum

export OMP_NUM_THREADS=1
srun -N 1 -n 1 -c 24 -p postproc --mem=12g --time=24:00:00 --account=msrad \
--job-name=$expname --output=$expname.out --error=$expname.err python $python_script &


