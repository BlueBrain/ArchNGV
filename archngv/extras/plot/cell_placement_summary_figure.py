import numpy as np
import pylab as plt
from .common import add_layers
from .common import remove_spines
from .pair_correlation import plot_pair_correlation
from .densities import plot_spatial_distribution_histogram
from .nearest_neighbors import plot_nearest_neighbor_distances
from .average_density_comparison import plot_laminar_density_comparison
from .radii_comparison import plot_radii_comparison

nn_literature = {'Lopez-Hidalgo et al., 2016': (30., 8.)}



literature_radii = {'value': [5.99, 4.36, 4.85, 5.9, 6.46, 6.05],
                    'citation': ['Puschmann et al., 2014',
                                 'Kali et al.',
                                 'Bindocci et al., 2017',
                                 'Lee et al., 2016',
                                 'Guo et al., 2016',
                                 'Bagheri et al., 2013']}

layers = None

def cell_placement_summary(placement_points,
                           placement_radii,
                           bounding_box,
                           densities_filename, figsize=(20, 20), neuronal_positions=None):


    #densities_filename = circuit.config.parameters['cell_placement']['laminar_densities']

    f = plt.figure(figsize=figsize)

    n_x = 3
    n_y = 3

    ax1 = plt.subplot2grid((n_x, n_y), (0, 0))
    ax2 = plt.subplot2grid((n_x, n_y), (0, 1))
    ax3 = plt.subplot2grid((n_x, n_y), (0, 2), rowspan=3)
    ax4 = plt.subplot2grid((n_x, n_y), (2, 0), colspan=2)
    ax5 = plt.subplot2grid((n_x, n_y), (1, 0))
    ax6 = plt.subplot2grid((n_x, n_y), (1, 1))
    #f, ax = plt.subplots(2, 2, figsize=figsize)

    placement_dataset = {'Simulation Result': placement_points}

    #######################################################################

    plot_nearest_neighbor_distances(ax1, placement_dataset, nn_literature)

    #######################################################################
    try:
        plot_pair_correlation(ax2, placement_points, bounding_box)
    except RuntimeError:
        pass
    #######################################################################

    plot_spatial_distribution_histogram(ax3, placement_points, densities_filename, bounding_box, neuronal_positions=neuronal_positions)

    #add_layers(ax3, layers, orientation='horizontal', color='k', alpha=0.6, linestyle='--')

    ax3.invert_xaxis()
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")


    remove_spines(ax3, (False,) * 4)

    plt_options = {'color': 'k', 'alpha':'0.6', 'linestyle': '--'}
    #add_layers(ax3, 'horizontal', 'left', plt_options)
    #######################################################################

    plot_laminar_density_comparison(ax4, placement_points, bounding_box)

    #######################################################################

    ax5.hist(placement_radii, bins=50, normed=True)
    remove_spines(ax5, (False, True, True, False))
    ax5.set_yticks([0., 0.5])
    ax5.set_xlim([3, 8])
    ax5.set_xticks([3.5, 5.5, 7.5])
    ax5.set_xlabel('Soma Radius (um)')
    ax5.set_ylabel('Density')

    #######################################################################

    plot_radii_comparison(ax6, placement_radii)

    return f, (ax1, ax2, ax3, ax4, ax5, ax6)
