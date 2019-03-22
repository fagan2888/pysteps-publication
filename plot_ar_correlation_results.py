
import pickle
from matplotlib import pyplot
import numpy as np

# the domain: "fmi" or "mch"
domain = "mch"

with open("ar2_corr_results_%s.dat" % domain, "rb") as f:
    results = pickle.load(f)

leadtimes = results["leadtimes"]
num_cascade_levels = len(results["cc_ar"])

fig = pyplot.figure(figsize=(5, 3.75))
ax = fig.gca()

for i in range(num_cascade_levels):
    ax.plot(leadtimes, results["cc_ar"][i] / results["n_ar_samples"][i], "k--")
    ax.plot(leadtimes, results["cc_obs"][i] / results["n_obs_samples"][i], "k-")

lines = ax.get_lines()
l = pyplot.legend([lines[0], lines[num_cascade_levels+1]], ["AR(2)", "Observed"],
                  fontsize=12, loc=(0.6, 0.9), framealpha=1.0)
ax.add_artist(l)

if domain == "mch":
    ax.text(155, 0.84, '1', fontsize=10)
    ax.text(139, 0.59, '2', fontsize=10)
    ax.text(120, 0.28, '3', fontsize=10)
    ax.text(88, 0.17, '4', fontsize=10)
    ax.text(63, 0.07, '5', fontsize=10)
    ax.text(38, 0.05, '6', fontsize=10)
    ax.text(15, 0.05, '7', fontsize=10)
    ax.text(7, 0.021, '8', fontsize=10)
else:
    ax.text(150, 0.85, '1', fontsize=10)
    ax.text(139, 0.71, '2', fontsize=10)
    ax.text(120, 0.35, '3', fontsize=10)
    ax.text(93, 0.18, '4', fontsize=10)
    ax.text(75, 0.07, '5', fontsize=10)
    ax.text(38, 0.05, '6', fontsize=10)
    ax.text(15, 0.05, '7', fontsize=10)
    ax.text(7, 0.021, '8', fontsize=10)

xt = np.hstack([[5], np.arange(0, np.max(leadtimes)+5, 20)])
ax.set_xticks(xt)
ax.set_xticklabels([int(v) for v in xt])
ax.tick_params(labelsize=10)

pyplot.grid(True)

pyplot.xlim(leadtimes[0], leadtimes[-1])
pyplot.ylim(-0.02, 1.1)

pyplot.xlabel("Lead time (minutes)", fontsize=12)
pyplot.ylabel("Correlation", fontsize=12)

pyplot.savefig("ar2_correlations_%s.pdf" % domain, bbox_inches="tight")
