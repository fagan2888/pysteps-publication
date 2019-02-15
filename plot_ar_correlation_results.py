
import pickle
from matplotlib import pyplot

# the domain: "fmi" or "mch"
domain = "fmi"

with open("ar2_corr_results_%s.dat" % domain, "rb") as f:
    results = pickle.load(f)

leadtimes = results["leadtimes"]
num_cascade_levels = len(results["cc_ar"])

fig = pyplot.figure()
ax = fig.gca()

for i in range(num_cascade_levels):
    ax.plot(leadtimes, results["cc_ar"][i] / results["n_ar_samples"][i], "k--")
    ax.plot(leadtimes, results["cc_obs"][i] / results["n_obs_samples"][i], "k-")

lines = ax.get_lines()
l = pyplot.legend([lines[0], lines[num_cascade_levels+1]], ["AR(2)", "Observed"],
                  fontsize=12)
ax.add_artist(l)

pyplot.grid(True)

pyplot.xlim(leadtimes[0], leadtimes[-1])
pyplot.ylim(-0.02, 1.15)

pyplot.xlabel("Lead time (minutes)", fontsize=12)
pyplot.ylabel("Correlation", fontsize=12)

pyplot.savefig("ar2_correlations.pdf", bbox_inches="tight")
