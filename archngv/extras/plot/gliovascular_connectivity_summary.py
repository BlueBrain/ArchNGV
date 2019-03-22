import numpy as np
import matplotlib.pylab as plt
from .common import remove_spines
from .common import add_layers
from .kernel_density import spatial_kernel_density_plot
from .cortical_depth_plot import cortical_depth_plot
from .separation import plot_laminar_separation
from .separation import  plot_average_separation_comparison
from .spatial_density_histogram import plot_laminar_mass_density, plot_laminar_endeet_contact_mass_density
from .unicode_maps import greek_alphabet_inverted, make_superscript
layers = np.array([0.0, 674.68206269999996, 1180.8844627000001, 1363.6375343, 1703.8656135000001, 1847.3347831999999, 2006.3482524000001])


def _create_subplot_grid():

    n_x, n_y = 4, 4

    ax1 = plt.subplot2grid((n_x, n_y), (0, 0), rowspan=2)
    remove_spines(ax1, (False, False, False, False))

    ax2 = plt.subplot2grid((n_x, n_y), (0, 1), rowspan=2)
    remove_spines(ax2, (False, False, False, False))

    ax3 = plt.subplot2grid((n_x, n_y), (0, 2), rowspan=2)
    remove_spines(ax3, (False, False, False, False))

    ax4 = plt.subplot2grid((n_x, n_y), (0, 3), rowspan=2)
    remove_spines(ax4, (False, False, False, False))

    ax5 = plt.subplot2grid((n_x, n_y), (2, 0), colspan=3)

    ax6 = plt.subplot2grid((n_x, n_y), (3, 0), colspan=3)

    ax7  = plt.subplot2grid((n_x, n_y), (2, 3))
    remove_spines(ax7, (False, True, True, False))

    ax8 = plt.subplot2grid((n_x, n_y), (3, 3))
    remove_spines(ax8, (False, True, True, False))

    return [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]


def plot_endfeet_per_astrocyte_histogram(axis, circuits, plt_options):

    number_of_endfeet = []

    for i, circuit in enumerate(circuits):

        surface_targeting_results = circuit.gliovascular_connectivity['surface_targeting']

        astro_idx, _ = surface_targeting_results['connectivity'].T

        number_of_endfeet.extend(np.bincount(astro_idx).tolist())

    axis.hist(number_of_endfeet, **plt_options)

    axis.set_xlim(min(number_of_endfeet) - 1, max(number_of_endfeet) + 1)


def plot_gliovascular_connectivity_summary(circuit, cutoff_radius=6., figsize=(20, 20),
                                           kernel_density_subsample_step=1):

    circuits = [circuit]

    f = plt.figure(figsize=figsize)
    axes = _create_subplot_grid()

    ngv_data = circuit.data

    astrocyte_positions = ngv_data.cells.astrocyte_positions
    astrocyte_radii = ngv_data.cells.astrocyte_radii

    vasculature = ngv_data.vasculature

    bounding_box = vasculature.bounding_box
    vasc_points = vasculature.points
    vasc_radii  = vasculature.radii
    seg_starts, seg_ends = vasculature.segments
    rad_starts, rad_ends = vasculature.segments_radii
    mean_radii = 0.5 * (rad_starts + rad_ends)

    seg_mask = mean_radii > cutoff_radius
    pnt_mask = vasc_radii > cutoff_radius

    x_range, y_range, z_range = vasculature.bounding_box.ranges.T

    graph_points = ngv_data.gliovascular.endfoot_graph_coordinates

    surface_targeting_results = ngv_data.gliovascular.endfoot_surface_coordinates

    ######################################################################################

    layer_options = {'color': 'w',
                     'alpha': 0.6,
                     'linestyle': '--',
                     'linewidth': 2}

    ax1 = axes[0]

    levels = spatial_kernel_density_plot(ax1, graph_points, x_range, y_range, 'Greys',
                                         subsample_step=kernel_density_subsample_step)

    #add_layers(ax1, 'horizontal', 'left', layer_options)
    ax1.set_title('Endfeet Contacts')
    ax1.set_xticks([])

    ######################################################################################

    ax2 = axes[1]

    levels = spatial_kernel_density_plot(ax2, vasc_points[pnt_mask], x_range, y_range, 'Reds',
                                         subsample_step=kernel_density_subsample_step)
    #add_layers(ax2, 'horizontal', 'left', layer_options)
    ax2.set_title('Large Vessels')
    ax2.set_xticks([])

    ######################################################################################

    ax3 = axes[2]

    levels = spatial_kernel_density_plot(ax3, vasc_points[~pnt_mask], x_range, y_range, 'Blues',
                                         subsample_step=kernel_density_subsample_step)
    #add_layers(ax3, 'horizontal', 'left', layer_options)
    ax3.set_title('Capillaries')
    ax3.set_xticks([])
    ######################################################################################

    ax4 = axes[3]

    levels = spatial_kernel_density_plot(ax4, astrocyte_positions, x_range, y_range, 'Greens',
                                         subsample_step=kernel_density_subsample_step)
    #add_layers(ax4, 'horizontal', 'left', layer_options)
    ax4.set_title('Astrocytic Somata')
    ax4.set_xticks([])

    ######################################################################################

    for ax in [ax1, ax2, ax3, ax4]:

        ax.set_ylim([layers[1] * 0.6, layers[-1]])

    ######################################################################################

    ax5 = axes[4]

    options = {'measurement_function': 'density',
               'orientation': 'horizontal',
               'xz_shape': 'rectangle',
               'n_bins': 30,
               'scale': 1e9}

    plt_options = {'color': 'k'}


    layer_options = {'color': 'k', 'alpha': 0.6, 'linestyle': '--'}


    ax5.set_yticks([10000, 20000, 30000])
    #add_layers(ax5, 'vertical', 'right', layer_options)
    plot_laminar_endeet_contact_mass_density(ax5, circuits, options, plt_options)

    remove_spines(ax5, (False,) * 4)

    ax5.set_xlim([layers[1] * 0.8, layers[-1]])
    ax5.set_ylim([10000, 30000])

    ylab = 'Endfeet Contact Density\n(Endfeet / mm{})'.format(make_superscript(3))
    ax5.set_ylabel(ylab)
    ax5.invert_xaxis()

    ######################################################################################

    ax6 = axes[5]

    options = {'orientation': 'horizontal', 'xz_shape': 'rectangle', 'n_bins': 30, 'scale': 1.}
    plt_options = {'color': 'k'}
    plot_laminar_separation(ax6, circuits, options,  plt_options)

    ylab = 'Astrocyte - Vessel\nSeparation ({}m)'.format(greek_alphabet_inverted['mu'])

    ax6.set_ylabel(ylab)

    ax6.set_xlim([layers[1] * 0.6, layers[-1]])

    remove_spines(ax6, (False,) * 4)

    ax6.set_yticks([5, 10, 15])


    #add_layers(ax6, 'vertical', 'right', layer_options)

    ax6.set_xlim([layers[1] * 0.8, layers[-1]])
    ax6.invert_xaxis()
    ######################################################################################

    ax7 = axes[6]

    plt_options = {'color': 'k', 'alpha': 0.8}
    plot_endfeet_per_astrocyte_histogram(ax7, circuits, plt_options)
    ax7.set_xlabel('Endfeet per Astrocyte')
    ax7.set_xticks([0, 1, 2])
    ax7.yaxis.set_visible(False)
    ax7.spines['left'].set_visible(False)

    ######################################################################################

    ax8 = axes[7]

    plt_options = {'alpha': 0.8}
    plot_average_separation_comparison(ax8, circuits, plt_options)
    #ax8.set_xlim([-0.05, 0.25])
    ax8.set_ylim([0, 15])
    ax8.set_xticks([])
    ax8.set_yticks([5, 10])

    ylab = 'Average Separation {}m'.format(greek_alphabet_inverted['mu'])
    ax8.set_ylabel(ylab)
    ax8.yaxis.tick_right()
    ax8.yaxis.set_label_position("right")
    remove_spines(ax8, (False, True, False, False))
    return f, axes
