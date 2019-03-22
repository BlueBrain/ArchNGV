import logging
import numpy as np
import pandas as pd
import seaborn as sns
from .common import bin_centers
from .common import remove_spines
from archngv.core.cell_placement.density import read_densities_from_file


log = logging.getLogger(__name__)


def _experimental_densities(densities_filename):

    log.info("Laminar Density File: {}".format(densities_filename))

    rdensities, rbins = read_densities_from_file(densities_filename)

    return rdensities, rbins, rbins[1] - rbins[0]


def _sim_densities_from_points(points, slab_volume, max_y, bins):

    # offset them from the max to reverse order
    y_values = max_y - points[:, 1]

    h, _ = np.histogram(y_values, bins)

    return h.astype(np.float) / slab_volume


def _calculate_rectangular_slab_volume(bounding_box, bin_size):
    x_range, _, z_range = bounding_box.ranges.T
    return 1e-9 * np.diff(x_range)[0] * np.diff(z_range)[0] * bin_size

def plot_spatial_distribution_histogram(ax, points, densities_filename, bounding_box, neuronal_positions=None):

    # get densities and bin data
    rdensities, bins, bin_size = _experimental_densities(densities_filename)

    x_range, y_range, z_range = bounding_box.ranges.T
    slab_volume = 1e-9 * np.diff(x_range)[0] * np.diff(z_range)[0] * bin_size

    max_y = y_range[-1]
    bin_starts = bins[:-1] # the last one is not a start

    sdensities = _sim_densities_from_points(points, slab_volume, max_y, bins)

    assert (len(bin_starts) == len(sdensities) == len(rdensities))

    max_bin = np.where(bin_starts > sdensities)[0][0]

    if neuronal_positions is not None:

        ndensities, nbins = _sim_densities_from_points(neuronal_positions, slab_volume, rdensities.size, dy)


        ax.barh(nbins[:-1], ndensities / 20., height=bin_width, alpha=0.2,
                label='Neuronal (downscaled)', color='k')

    max_bin = np.where(bin_starts > sdensities)[0][0]

    ax.barh(bin_starts[:max_bin],
            rdensities[:max_bin],
            height=bin_size,
            alpha=0.8,
            align='edge',
            label='Appaix et al., 2012',
            color='cornflowerblue')

    ax.barh(bin_starts,
            sdensities,
            height=bin_size,
            alpha=0.8,
            align='edge',
            color='darkred',
            label='Simulation Result')

    #ax.set_ylabel("Cortical Depth (um)")
    ax.set_xlabel("Astrocyte Density\n(astrocytes / cubic mm)")

    ax.set_xticks([0, 20000])
    ax.set_xlim([0., 20000])

    ax.set_ylim([0, 1500])


    ax.invert_yaxis()
    ax.legend(loc=2)
