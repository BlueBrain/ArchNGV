import numpy as np
import logging
import seaborn as sns
from itertools import cycle
from .common import bin_centers
from .common import plot_oriented
from .common import smooth
from .common import LAYERS
from archngv.extras.analysis.histograms import wiring_histogram
from archngv.extras.analysis.histograms import vectors_included_in_bin_1D
from archngv.extras.analysis.histograms import points_included_in_bin_1D
from morphmath import normalize_vectors

from morphspatial.shapes import Sphere

L = logging.getLogger(__name__)


def account_for_radii(astro_positions, astro_radii, target_positions):

    dirs = target_positions - astro_positions

    u_dirs = normalize_vectors(dirs)
    new_starts = astro_positions + u_dirs * astro_radii[:, np.newaxis]
    return new_starts


def astrocyte_capillary_vectors(astro_positions, astro_radii, surface_targeting_results, idx_to_ignore=None):

    connectivity = surface_targeting_results['connectivity'].copy()

    if idx_to_ignore is not None:

        to_remove = np.where(np.in1d(connectivity[:, 0], idx_to_ignore))[0]

        connectivity = np.delete(connectivity, to_remove, axis=0)

    astro_idx, target_idx = connectivity.T

    target_positions = surface_targeting_results['points'][target_idx]

    astro_positions = account_for_radii(astro_positions[astro_idx],
                                        astro_radii[astro_idx],
                                        target_positions)

    return astro_positions, target_positions


def laminar_astrocyte_capillary_separation(astro_positions, astro_radii, bounding_box, surface_targeting_results):

    vec_starts, vec_ends = astrocyte_capillary_vectors(astro_positions, astro_radii, surface_targeting_results)

    options = {'measurement_function': 'mean', 'xz_shape': 'rectangular', 'n_bins': 30}

    bins, measurements = wiring_histogram(vec_starts, vec_ends, bounding_box, options)

    bin_cents = bin_centers(bins)

    return bin_cents, measurements


def sphere_index_separation(center, radius, index):

    vdata = index.nearest_neighbors(center, radius, n_neighbors=1)[0]
    vessel_radius = vdata[3]
    return np.linalg.norm(vdata[:3] - center) - radius - vessel_radius


def vasculature_nearest_sphere(center, radius, index):
    sphere = Sphere(center, radius)
    near_sphere = next(index.nearest_neighbors(sphere, n_neighbors=1))

    return near_sphere.center.tolist() + near_sphere.radius

def data_from_circuit(circuit, bounding_box):

    ngv_data = circuit.data

    astro_pos = ngv_data.cells.astrocyte_positions[:]
    astro_rad = ngv_data.cells.astrocyte_radii[:]

    if bounding_box is not None:

        mask = bounding_box.spheres_inside(astro_pos, astro_rad)

        astro_pos = astro_pos[mask]
        astro_rad = astro_rad[mask]

    index_path = circuit.config.output_paths('vasculature_index')
    index = SphereSpatialIndex.load(index_path)

    return astro_pos, astro_rad, index


def plot_laminar_separation(axis, circuits, options, plt_options):

    separations_per_circuit = []
    options = {'measurement_function': 'mean', 'xz_shape': 'rectangular', 'n_bins': 30}
    for i, circuit in enumerate(circuits):

        bounding_box = circuit.data.vasculature.bounding_box

        astro_positions, astro_radii, index = data_from_circuit(circuit, bounding_box)

        vasc_data = np.vstack([vasculature_nearest_sphere(p, r, index) for (p, r) in zip(astro_positions, astro_radii)])

        vasculature_positions = vasc_data[:, :3]
        vasculature_radii = vasc_data[:, 3]

        vectors = vasculature_positions - astro_positions

        distances = np.linalg.norm(vectors, axis=1)

        u_vectors = vectors / distances[:, np.newaxis]

        a_periphery_points = astro_positions + u_vectors * astro_radii[:, np.newaxis]
        v_periphery_points = vasculature_positions - u_vectors * vasculature_radii[:, np.newaxis]


        bins, measurements = wiring_histogram(a_periphery_points, v_periphery_points, bounding_box, options)

        bin_cents = bin_centers(bins)

        axis.plot(bin_cents, measurements, color='gray', linewidth=0.5)

        separations_per_circuit.append(measurements)

    if len(circuits) > 1:

        mean_separations = np.mean(separations_per_circuit, axis=0)
        sdev_separations = np.std(separations_per_circuit, axis=0)

        #N = separations.size * 2

        axis.plot(bin_cents, mean_separations, color='k', linewidth=1.5)
        axis.fill_between(bin_cents, mean_separations - sdev_separations, mean_separations + sdev_separations, color='gray', alpha=0.6)


def plot_separation_distribution(axis, circuit, plt_options, bounding_box=None):

    astro_pos, astro_rad, index = data_from_circuit(circuit, bounding_box)

    separations = np.fromiter((sphere_index_separation(p, r, index) for (p, r) in zip(astro_pos, astro_rad)),
                              dtype=np.float)

    axis.hist(separations, **plt_options)


def plot_separation_distribution2(axis, circuit, plt_options, bounding_box=None):

    astro_pos = circuit.astrocyte_positions
    astro_rad = circuit.astrocyte_radii

    surface_targeting_results = circuit.gliovascular_connectivity['surface_targeting']

    if bounding_box is not None:

        mask = bounding_box.spheres_inside(astro_pos, astro_rad)
        idx = np.where(~mask)[0]

    vec_starts, vec_ends = astrocyte_capillary_vectors(astro_pos, astro_rad,
                                                       surface_targeting_results, idx_to_ignore=idx)

    vectors = vec_ends - vec_starts

    distances = np.linalg.norm(vectors, axis=1)

    axis.hist(distances, **plt_options)


def plot_per_layer_separation_distribution(axes, circuit, plt_options, bounding_box=None):

    assert len(axes) == 6

    astro_pos, astro_rad, index = data_from_circuit(circuit, bounding_box)

    layers = LAYERS['bins']
    labels = LAYERS['labels']

    for i, axis in enumerate(axes):

        label = labels[i]
        mask = points_included_in_bin_1D(astro_pos, layers[i], layers[i + 1])

        masked_pos = astro_pos[mask]
        masked_rad = astro_rad[mask]

        distances = np.fromiter((sphere_index_separation(p, r, index) for (p, r) in zip(masked_pos, masked_rad)),
                              dtype=np.float)

        axis.hist(distances, **plt_options)

        axis.text(0.95, 0.5, "Layer {}".format(label),
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axis.transAxes)

        axis.legend()


def plot_per_layer_astrocyte_endfeet_separation(axes, circuit, plt_options):

    assert len(axes) == 6

    astro_pos = circuit.astrocyte_positions
    astro_rad = circuit.astrocyte_radii

    surface_targeting_results = circuit.gliovascular_connectivity['surface_targeting']

    vec_starts, vec_ends = astrocyte_capillary_vectors(astro_pos, astro_rad, surface_targeting_results)

    layers = LAYERS['bins']
    labels = LAYERS['labels']

    for i, axis in enumerate(axes[::-1]):

        label = labels[i]

        mask = vectors_included_in_bin_1D(vec_starts, vec_ends, layers[i], layers[i + 1])

        distances = np.linalg.norm(vec_ends[mask] - vec_starts[mask], axis=1)

        axis.hist(distances, **plt_options)
        """
        axis.text(0.95, 0.5, "Layer {}".format(label),
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axis.transAxes)
        """
        axis.set_title("Layer {}".format(label))

def plot_separation(axis, circuits, options,  plt_options):

    try:

        _plot_separation_multip(axis, circuits, options, plt_options)

    except TypeError:

        L.info('Only one circuit')
        _plot_separation_single(axis, circuits, options, plt_options)




def _plot_separation_single(axis, circuit):


    astro_positions = circuit.astrocyte_positions
    astro_radii = circuit.astrocyte_radii

    bounding_box = circuit.vasculature.bounding_box

    surface_targeting_results = circuit.gliovascular_connectivity['surface_targeting']


    bin_centers, separations = laminar_astrocyte_capillary_separation(astro_positions,
                                                                      astro_radii,
                                                                      bounding_box,
                                                                      surface_targeting_results)

    scale = options['scale']

    N = separations.size * 2

    y, x = smooth(scale * separations, bin_centers, N)

    axis.plot(x, y, **plt_options)


def _plot_separation_multip(axis, circuits, options, plt_options):

    all_separations = []

    for i, circuit in enumerate(circuits):

        astro_positions = circuit.astrocyte_positions
        astro_radii = circuit.astrocyte_radii

        bounding_box = circuit.vasculature.bounding_box

        vasculature_nearest_sphere()
        bin_centers, separations = laminar_astrocyte_capillary_separation(astro_positions,
                                                                          astro_radii,
                                                                          bounding_box,
                                                                          surface_targeting_results)


        axis.plot(bin_centers, separations, color='gray', linewidth=0.5)

        all_separations.append(separations)



        L.info("Separations for circuit {} calculated.".format(i))

    mean_separations = np.mean(all_separations, axis=0)
    sdev_separations = np.std(all_separations, axis=0)
    scale = options['scale']

    N = separations.size * 2

    #y_mean, x_mean = smooth(scale * mean_separations, bin_centers, N)
    #y_sdev, x_sdev = smooth(scale * sdev_separations, bin_centers, N)

    axis.plot(bin_centers, mean_separations, color='k', linewidth=1.5)
    axis.fill_between(bin_centers, mean_separations - sdev_separations, mean_separations + sdev_separations, color='gray', alpha=0.6)


def plot_average_separation_comparison(axis, circuits, plt_options):

    sdev_separations = np.zeros(len(circuits), dtype=np.float)
    mean_separations = np.zeros(len(circuits), dtype=np.float)

    for i, circuit in enumerate(circuits):

        L.info('Calc separation for {}-th circuit'.format(i))

        astro_positions, astro_radii, index = data_from_circuit(circuit, circuit.vasculature.bounding_box)
        surface_targeting_results = circuit.gliovascular_connectivity['surface_targeting']

        distances = np.fromiter((sphere_index_separation(p, r, index) for (p, r) in zip(astro_positions, astro_radii)),
                                 dtype=np.float)

        mean_separations[i] = distances.mean()

    mean_separation = mean_separations.mean()
    sdev_separation = mean_separations.std()


    data = {'mean': [mean_separation, 8.],
            'sdev': [sdev_separation, 2.],
            'citation': ['Simulation Result', 'McCaslin et al., 2011']}

    bar_width = 0.1
    error_config = {'ecolor': '0.3'}

    spacings = 0.5 * np.arange(len(data['citation']))

    palette = cycle(sns.color_palette())

    for i, (m, s, cit) in enumerate(zip(data['mean'], data['sdev'], data['citation'])):

        if cit == 'Simulation Result':
            color = 'k'

        else:
            color = next(palette)

        axis.bar(spacings[i], m, bar_width, yerr=s, color=color, label=cit, error_kw=error_config,  **plt_options)

    axis.set_ylabel('Average Separation')

    axis.set_xlim(-0.5, spacings[-1] + 0.5)
    axis.legend()
