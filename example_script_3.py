# Listing 3

from pysteps.postprocessing import ensemblestats
from pysteps.utils import conversion
from pysteps import verification

# find the files containing the verifying observations
fns = io.archive.find_by_date(date, root_path, "%Y%m%d", fn_pattern, fn_ext,
                              5, 0, num_next_files=12)

# read the observations
Z_obs, _, metadata = io.read_timeseries(fns, import_fmi_pgm, gzipped=True,
                                        num_next_files=12)
R_obs = conversion.to_rainrate(Z_obs, metadata, 223.0, 1.53)[0]

# iterate over the nowcast lead times
for lt in range(R_f.shape[1]):
  # compute the exceedance probability of 0.1 mm/h from the ensemble
  P_f = ensemblestats.excprob(R_f[:, lt, :, :], 0.1, ignore_nan=True)

  # compute and plot the ROC curve
  roc = verification.ROC_curve_init(0.1, n_prob_thrs=10)
  verification.ROC_curve_accum(roc, P_f, R_obs[lt+1, :, :])
  fig = figure()
  verification.plot_ROC(roc, ax=fig.gca(), opt_prob_thr=True)
  pyplot.savefig("ROC_%02d.eps" % (lt+1), bbox_inches="tight")
  pyplot.close()

  # compute and plot the reliability diagram
  reldiag = verification.reldiag_init(0.1)
  verification.reldiag_accum(reldiag, P_f, R_obs[lt+1, :, :])
  fig = figure()
  verification.plot_reldiag(reldiag, ax=fig.gca())
  pyplot.savefig("reldiag_%02d.eps" % (lt+1), bbox_inches="tight")
  pyplot.close()

  # compute and plot the rank histogram
  rankhist = verification.rankhist_init(R_f.shape[0], 0.1)
  verification.rankhist_accum(rankhist, R_f[:, lt, :, :], R_obs[lt+1, :, :])
  fig = figure()
  verification.plot_rankhist(rankhist, ax=fig.gca())
  pyplot.savefig("rankhist_%02d.eps" % (lt+1), bbox_inches="tight")
  pyplot.close()
