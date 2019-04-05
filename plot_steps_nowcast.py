# Computes and plots a STEPS nowcast (Figures A1 and A2 in the paper).

from pylab import *
from datetime import datetime
from pysteps.io.archive import find_by_date
from pysteps.io.importers import import_fmi_pgm, import_mch_gif
from pysteps.io.readers import read_timeseries
from pysteps.motion.darts import DARTS
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import nowcasts
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, transformation
from pysteps.utils.dimension import square_domain
from pysteps.visualization import plot_precip_field, quiver
from pysteps.visualization.utils import parse_proj4_string

date = datetime.strptime("201609281600", "%Y%m%d%H%M")
root_path = "/top/college/academic/ECE/spulkkin/home/ohjelmistokehitys/pySTEPS/pysteps-data/radar/fmi/20160928"
path_fmt = ""
fn_pattern = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
fn_ext = "pgm.gz"
timestep = 5
n_ens_members = 24
num_workers = 12
seed = 24
map_plotter = "basemap"
basemap_resolution = 'h'

inputfns = find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep, 
                        num_prev_files=9)

Z, _ ,metadata = read_timeseries(inputfns, import_fmi_pgm, gzipped=True)

R = conversion.to_rainrate(Z, metadata, 223.0, 1.53)[0]
R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
R[~np.isfinite(R)] = -15.0

V = dense_lucaskanade(R)

# the S-PROG nowcast
nowcast_method = nowcasts.get_method("sprog")
R_f = nowcast_method(R[-3:, :, :], V, 12, n_cascade_levels=8,
                     R_thr=-10.0, decomp_method="fft",                     
                     bandpass_filter_method="gaussian",
                     probmatching_method="mean", fft_method="pyfftw")

R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

fig = figure(figsize=(9, 9))
ax = fig.add_subplot(221)
ax.set_title("a)", loc="left", fontsize=12)

if map_plotter == "cartopy":
    plot_precip_field(R_f[-1, :, :], map="cartopy", geodata=metadata, 
                      drawlonlatlines=True, cartopy_scale="50m")
else:
    bm = plot_precip_field(R_f[-1, :, :], map="basemap", geodata=metadata,
                           drawlonlatlines=False, basemap_resolution=basemap_resolution,
                           basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120])

# the STEPS nowcast
nowcast_method = nowcasts.get_method("steps")
R_f = nowcast_method(R[-3:, :, :], V, 12, n_ens_members=n_ens_members,
                     n_cascade_levels=8, R_thr=-10.0,
                     kmperpixel=1.0, timestep=5, decomp_method="fft",
                     bandpass_filter_method="gaussian",
                     noise_method="nonparametric", num_workers=num_workers,
                     vel_pert_method="bps", mask_method="incremental",
                     fft_method="pyfftw", seed=seed)

R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

ax = fig.add_subplot(222)
ax.set_title("b)", loc="left", fontsize=12)

R_f_mean = np.mean(R_f[:, -1, :, :], axis=0)

if map_plotter == "cartopy":
    plot_precip_field(R_f_mean, map="cartopy", geodata=metadata,
                      drawlonlatlines=True, cartopy_scale="50m")
else:
    bm = plot_precip_field(R_f_mean, map="basemap", geodata=metadata,
                           drawlonlatlines=False, basemap_resolution=basemap_resolution,
                           basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120])

for i in range(2):
  ax = fig.add_subplot(223 + i)
  if i == 0:
      ax.set_title("c)", loc="left", fontsize=12)
  else:
      ax.set_title("d)", loc="left", fontsize=12)

  if map_plotter == "cartopy":
      plot_precip_field(R_f[i, -1, :, :], map="cartopy", geodata=metadata,
                        drawlonlatlines=True, cartopy_scale="50m")
  else:
      bm = plot_precip_field(R_f[i, -1, :, :], map="basemap", geodata=metadata,
                             drawlonlatlines=False, basemap_resolution=basemap_resolution,
                             basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120])

subplots_adjust(wspace=-0.15, hspace=0.1)

savefig("steps_nowcast_examples.png", bbox_inches="tight", dpi=200)

P = excprob(R_f[:, -1, :, :], 0.5)

figure()

if map_plotter == "cartopy":
    plot_precip_field(P, map="cartopy", geodata=metadata, drawlonlatlines=True,
                      cartopy_scale="50m", type="prob", units="mm/h", probthr=0.5)
else:
    bm = plot_precip_field(P, map="basemap", geodata=metadata, drawlonlatlines=False,
                           basemap_resolution=basemap_resolution,
                           basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120],
                           type="prob", units="mm/h", probthr=0.5)

savefig("steps_excprobs.png", bbox_inches="tight", dpi=200)
