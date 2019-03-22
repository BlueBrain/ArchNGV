import numpy as np
import pylab as plt
from .common import remove_spines
from ..analysis.nearest_neighbor import nearest_neighbor_distances


def plot_nearest_neighbor_distances(ax, datasets_to_plot, literature_data):

    # the datasets of points to be histed
    for label, points in datasets_to_plot.items():

        values = nearest_neighbor_distances(points, points)

        ax.hist(values, bins=50, alpha=0.8, normed=True, label=label, histtype='bar')

    # literature dataset distributions
    for source, values in literature_data.items():

        source_mean, source_std = values

        x = np.linspace(source_mean - 3. * source_std, source_mean + 3. * source_std, 1000)

        ax.plot(x, plt.normpdf(x, source_mean, source_std), label=source, alpha=0.8, lw=2)


    # some formatting
    remove_spines(ax, (False, True, True, False))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks([0., 30., 60.])
    ax.set_xlim([0., 60.])
    ax.set_yticks([0., 0.05])
    ax.set_ylim([0., 0.08])
    ax.set_xlabel('Nearest Neighbor Distance (um)')
    ax.set_ylabel('Density')
    ax.legend(loc=2, frameon=False)


def plot_nearest_neighbor_distances_circuit(axis, circuit, plt_options):

    positions = circuit.astrocyte_positions

    values = nearest_neighbor_distances(positions, positions)

    axis.hist(values, **plt_options)

