# Plots the results produced by run_optflow comparison.py (Figure 12 in the 
# paper).
import numpy
import pickle
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from os.path import join
from string import ascii_lowercase

if __name__ == "__main__":

    domains = ['mch', 'us']
    domains_data_dirs = {'us': './data/optflow_comparison/us',
                         'mch': './data/optflow_comparison/mch'}

    domains_labels = {'us': "US Composites",
                      'mch': 'MeteoSwiss'}

    # nowcast methods to include in the plot
    nowcast_methods = ["advection", "sprog"]

    nowcast_methods_label = {"advection": "Advection",
                             "sprog": "S-PROG"}

    # line and marker parameters
    linecolors = {"darts": "#1f77b4",
                  "lucaskanade": "#ff7f0e",
                  "vet": "#2ca02c"}
    # legend labels corresponding to the optical flow methods
    optflow_method_labels = {"darts": "DARTS",
                             "lucaskanade": "Lucas-Kanade",
                             "vet": "VET"}

    markers = {"advection": 'o',
               "sprog": None}

    metrics_units = {'CSI': "", 'MAE': ' [mm/h]'}
    # minimum lead time to plot (minutes)
    minleadtime = 5

    # maximum lead time to plot (minutes)
    maxleadtime = 120
    scale = 1.1
    pyplot.figure(figsize=(8*scale, 5.5*scale))

    i = 0
    for metric in ['CSI', 'MAE']:
        for _domain in domains:

            data_dir = domains_data_dirs[_domain]

            file_name_pattern = join(data_dir,
                                     "optflow_comparison_results_{}_{}.dat")

            ax = pyplot.subplot(221 + i)

            for ncm in nowcast_methods:

                file_name = file_name_pattern.format(ncm, _domain)

                with open(file_name, "rb") as f:
                    results = pickle.load(f)

                oflow_methods = sorted(results.keys())

                for oflow_method in oflow_methods:
                    n_samples = numpy.asarray(
                        results[oflow_method]["n_samples"])

                    ncm_ = "Advection" if ncm == "advection" else "S-PROG"

                    label = '{}/{}'.format(
                        nowcast_methods_label[ncm],
                        optflow_method_labels[oflow_method])

                    metric_data = numpy.asarray(
                        results[oflow_method][metric]) / n_samples

                    leadtimes = (numpy.arange(len(metric_data)) + 1) * 5

                    ax.plot(leadtimes, metric_data,
                            color=linecolors[oflow_method],
                            marker=markers[ncm],
                            label=label,
                            linewidth=2,
                            markersize=6)

                if i == 1:
                    ax.legend(fontsize=10,
                              labelspacing=0.4,
                              loc='upper right',
                              framealpha=1.,
                              facecolor='white',
                              bbox_to_anchor=[1.01, 1.02])

                if i > 1:
                    ax.set_xlabel("Forecast time [minutes]")
                ax.set_ylabel(metric + metrics_units[metric])
                ax.grid(True)
                ax.set_xlim(minleadtime, maxleadtime)
                ax.xaxis.set_major_locator(MultipleLocator(20))

                if metric == 'CSI':
                    ax.yaxis.set_major_locator(MultipleLocator(0.1))
                    # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    ax.set_ylim(0.31, 0.9)
                else:
                    ax.yaxis.set_major_locator(MultipleLocator(0.5))
                    # ax.set_ylim(top=numpy.max(metric_data*1.05))

                ax.text(0., 1.02, ascii_lowercase[i] + ")",
                        transform=ax.transAxes)

                if i < 2:
                    ax.set_title("{} Events".format(domains_labels[_domain]),
                                 fontsize=14, pad=10)

            i += 1

    pyplot.subplots_adjust(hspace=0.18, wspace=0.2,
                           top=0.93,
                           left=0.08, bottom=0.09, right=0.97)

    pyplot.savefig("oflow_benchmark_summary.pdf")
