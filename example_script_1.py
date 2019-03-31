# Listing 1

from datetime import datetime
from matplotlib import cm, pyplot
import numpy as np
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import io
from pysteps.io.importers import import_fmi_pgm
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils import conversion, transformation

date       = datetime.strptime("201609281600", "%Y%m%d%H%M")
#root_path  = "/home/pysteps-data/radar/fmi"
root_path  = "/top/college/academic/ECE/spulkkin/home/ohjelmistokehitys/pySTEPS/pysteps-data/radar/fmi"
fn_pattern = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
fn_ext     = "pgm.gz"

# find the input files from the archive
fns = io.archive.find_by_date(date, root_path, "%Y%m%d", fn_pattern, fn_ext, 5, 
                              num_prev_files=9)

# read the radar composites and apply thresholding
Z, _, metadata = io.read_timeseries(fns, import_fmi_pgm, gzipped=True)
R = conversion.to_rainrate(Z, metadata, 223.0, 1.53)[0]
R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
R[~np.isfinite(R)] = -15.0

# construct bandpass filter and apply the cascade decomposition
filter = filter_gaussian(R.shape[1:], 7)
decomp = decomposition_fft(R[-1, :, :], filter)

# plot the normalized cascade levels
for i in range(7):
    mu = decomp["means"][i]
    sigma = decomp["stds"][i]
    decomp["cascade_levels"][i] = (decomp["cascade_levels"][i] - mu) / sigma

fig, ax = pyplot.subplots(nrows=2, ncols=4)

ax[0, 0].imshow(R[-1, :, :], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 1].imshow(decomp["cascade_levels"][0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(decomp["cascade_levels"][1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 3].imshow(decomp["cascade_levels"][2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 0].imshow(decomp["cascade_levels"][3], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(decomp["cascade_levels"][4], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(decomp["cascade_levels"][5], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 3].imshow(decomp["cascade_levels"][6], cmap=cm.RdBu_r, vmin=-3, vmax=3)

ax[0, 0].set_title("Observed")
ax[0, 1].set_title("Level 1")
ax[0, 2].set_title("Level 2")
ax[0, 3].set_title("Level 3")
ax[1, 0].set_title("Level 4")
ax[1, 1].set_title("Level 5")
ax[1, 2].set_title("Level 6")
ax[1, 3].set_title("Level 7")

for i in range(2):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

pyplot.savefig("cascade_decomp.png", dpi=300, bbox_inches="tight")
