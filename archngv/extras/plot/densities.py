import logging
import numpy as np
from archngv.extras.analysis.density import read_densities_from_file


L = logging.getLogger(__name__)


def _point_histogram_across_y(points, bins, slab_volume):

    h, _ = np.histogram(points[:, 1], bins)
    return h.astype(np.float) / slab_volume


def _voxel_densities(voxel_data):

    y_bin_size = voxel_data.voxel_dimensions[1]
    n_x, n_y, n_z = voxel_data.shape
    xz_slab_volume = 1e-9 * voxel_data.voxel_volume * n_x * n_z

    y_bins = np.arange(0., (n_y + 1) * y_bin_size, y_bin_size) + voxel_data.offset[1]
    y_densities = np.mean(voxel_data.raw, axis=(0, 2))

    return y_densities, y_bins, y_bin_size, xz_slab_volume


def plot_spatial_distribution_histogram(ax, ref_densities, bounding_box, point_populations_dict, **kwargs):

    voxel_data = ref_densities['voxel_data']

    rdensities, bins, bin_size, slab_volume = _voxel_densities(voxel_data)
    bin_centers = bins[:-1] + bin_size * 0.5

    mask = ~np.isclose(rdensities, 0.0)
    ax.plot(rdensities[mask], bin_centers[mask], color=ref_densities['color'], linewidth=2, label=ref_densities['label'])

    ids = np.arange(0, len(bins), 3, dtype=np.intp)

    bins = np.take(bins, ids)
    rdensities = np.take(rdensities, ids[:-1])

    new_bin_size = bins[1] - bins[0]
    slab_volume = slab_volume * new_bin_size / bin_size

    _, y_range, _ = voxel_data.bbox.T

    for label, data in point_populations_dict.items():

        densities = _point_histogram_across_y(data['points'], bins, slab_volume)
        ax.barh(bins[:-1], densities, height=new_bin_size,
                align='edge', color=data['color'], label=label, linewidth=0.5, edgecolor=data['edgecolor'], **kwargs)
