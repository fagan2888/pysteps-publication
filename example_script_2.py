# Listing 2

from matplotlib import pyplot
from pysteps import motion, nowcasts
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import transformation
from pysteps.visualization import plot_precip_field, quiver

# compute the advection field
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R)

# compute the S-PROG nowcast
nowcast_method = nowcasts.get_method("sprog")
R_f_sprog = nowcast_method(R[-3:, :, :], V, 12, R_thr=-10.0)[-1, :, :]

# compute the STEPS nowcast
nowcast_method = nowcasts.get_method("steps")
R_f = nowcast_method(R[-3:, :, :], V, 12, n_ens_members=24, n_cascade_levels=8, 
                     R_thr=-10.0, kmperpixel=1.0, timestep=5)

# plot the S-PROG nowcast, one ensemble member of the STEPS nowcast and the exceedance
# probability of 0.1 mm/h computed from the ensemble
R_f_sprog = transformation.dB_transform(R_f_sprog, threshold=-10.0, inverse=True)[0]
pyplot.figure()
plot_precip_field(R_f_sprog, map="basemap", geodata=metadata, drawlonlatlines=True,
                  basemap_resolution='h')
pyplot.savefig("SPROG_nowcast.png", bbox_inches="tight", dpi=300)

R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

R_f_mean = np.mean(R_f[:, -1, :, :], axis=0)

pyplot.figure()
plot_precip_field(R_f_mean, map="basemap", geodata=metadata, drawlonlatlines=True,
                  basemap_resolution='h')
pyplot.savefig("STEPS_ensemble_mean.png", bbox_inches="tight", dpi=300)

pyplot.figure()
plot_precip_field(R_f[0, -1, :, :], map="basemap", geodata=metadata, drawlonlatlines=True,
                  basemap_resolution='h')
pyplot.savefig("STEPS_ensemble_member.png", bbox_inches="tight", dpi=300)

pyplot.figure()
P = excprob(R_f[:, -1, :, :], 0.5)
plot_precip_field(P, map="basemap", geodata=metadata, drawlonlatlines=True,
                  basemap_resolution='h', type="prob", units="mm/h", probthr=0.5)
pyplot.savefig("STEPS_excprob_0.5.png", bbox_inches="tight", dpi=300)
