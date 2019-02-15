
import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

from matplotlib import pyplot, ticker
import numpy as np
from scipy.interpolate import interp1d
from pysteps.cascade.bandpass_filters import filter_gaussian

grid_size = 1227
grid_res = 1.0
num_levels = 8
normalize = True

F = filter_gaussian(grid_size, num_levels)

fig = pyplot.figure()
ax1 = fig.gca()
ax2 = ax1.twiny()

w = F["weights_1d"]
cf = F["central_freqs"]

for i in range(len(w)):
    x_ip = np.linspace(0, len(w[i])-1, 10000)
    y_ip = interp1d(np.arange(len(w[i])), w[i], kind="cubic")(x_ip)
    ax1.semilogx(x_ip, y_ip, "k-", lw=2)
    ax1.plot([cf[i], cf[i]], [0, 1], "k--")

cf[0] = 1
xtl1 = ["%.1f" % (grid_size*grid_res)]
xtl2 = [0]
for v in cf[1:]:
  xtl1.append("%.2f" % (0.5*grid_size / v * grid_res))
  xtl2.append("%.2f" % v)

ax1.set_xlim(1, int(grid_size/2)+1)
ax1.set_ylim(0, 1)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax1.minorticks_off()
ax1.set_xticks(cf)
ax1.set_xticklabels(xtl2, fontsize=10)
ax1.set_xlabel(r"Radial wavenumber $|\boldsymbol{k}|$", fontsize=12)
ax1.set_ylabel("Normalized weight", fontsize=12)

ax2.set_xscale("log")
ax2.set_xlim(1, int(grid_size/2)+1)
ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.minorticks_off()
ax2.set_xticks(cf)
ax2.set_xticklabels(xtl1, fontsize=10)
ax2.set_xlabel("Spatial scale (kilometers)", fontsize=12)

fig.savefig("bandpass_filter_weights.pdf", bbox_inches="tight")
