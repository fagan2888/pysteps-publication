# Plots cascade decomposition of a reflectivity field (Figure 2 in the paper).

from datetime import datetime
from matplotlib import cm, pyplot
import numpy as np
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import io
from pysteps.io.importers import import_fmi_pgm
from pysteps.cascade.decomposition import decomposition_fft

date = datetime.strptime("201609281600", "%Y%m%d%H%M")
# insert your data path here
root_path = ""
fn_pattern = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
fn_ext = "pgm.gz"

cmap = cm.RdBu_r
vmin = -3
vmax = 3

fn = io.archive.find_by_date(date, root_path, "%Y%m%d", fn_pattern, fn_ext, 5)

R, _, metadata = io.read_timeseries(fn, import_fmi_pgm, gzipped=True)
R = R.squeeze()

R[R < 10.0] = 5.0
R[~np.isfinite(R)] = 5.0
R = (R - np.mean(R)) / np.std(R)

filter = filter_gaussian(R.shape, 8)
decomp = decomposition_fft(R, filter)

for i in range(8):
    mu = decomp["means"][i]
    sigma = decomp["stds"][i]
    decomp["cascade_levels"][i] = (decomp["cascade_levels"][i] - mu) / sigma

fig, ax = pyplot.subplots(nrows=2, ncols=4)

im = ax[0, 0].imshow(R, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 1].imshow(decomp["cascade_levels"][0], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[0, 2].imshow(decomp["cascade_levels"][1], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[0, 3].imshow(decomp["cascade_levels"][2], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[1, 0].imshow(decomp["cascade_levels"][3], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[1, 1].imshow(decomp["cascade_levels"][4], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[1, 2].imshow(decomp["cascade_levels"][5], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
ax[1, 3].imshow(decomp["cascade_levels"][6], cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)

scales = 1226.0 / (2*np.array(filter["central_freqs"]))
scales[0] = 1226.0

ax[0, 0].set_title("Observed", fontsize=12)
ax[0, 1].set_title("Level 1", fontsize=12)
ax[0, 2].set_title("Level 2", fontsize=12)
ax[0, 3].set_title("Level 3", fontsize=12)
ax[1, 0].set_title("Level 4", fontsize=12)
ax[1, 1].set_title("Level 5", fontsize=12)
ax[1, 2].set_title("Level 6", fontsize=12)
ax[1, 3].set_title("Level 7", fontsize=12)

for i in range(2):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
fig.colorbar(im, cax=cbar_ax)

pyplot.savefig("cascade_decomp.pdf", dpi=200, bbox_inches="tight")
