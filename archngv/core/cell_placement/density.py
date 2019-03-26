""" Voxeldata debsity creation and loading
"""

import numpy as np
from voxcell import VoxelData


def read_densities_from_file(filename):
    """
    Reads in the densities from file. It is assumed that the bins
    are equidistant, thus the depth of the circuit is used to
    derive their size.
    """
    # densities = np.tile(10700., 100)
    # assume they are in mm^3, thus they are converted to um^3
    densities, bins_left, bins_right = np.loadtxt(filename).T

    # bin size
    # dy = np.mean(bins_right - bins_left)

    bins = np.hstack((bins_left, bins_right[-1]))

    return densities, bins


def create_density_from_laminar_densities(laminar_densities, bins, bounding_box, n_x, n_z):
    ''' Converts the 1D laminar_densities to an equivalent 3D density grid
    '''
    laminar_densities = laminar_densities[::-1]

    n_bins_data = laminar_densities.size

    e_x, e_y, e_z = bounding_box.extent

    voxel_dimensions = (e_x / float(n_x), bins[1] - bins[0], e_z / float(n_z))

    n_ygrid = int(np.ceil(e_y / voxel_dimensions[1]))

    dgrid = np.zeros((n_x, n_ygrid, n_z), dtype=np.float)

    if n_bins_data > n_ygrid:

        nmin = n_bins_data - n_ygrid

        dgrid[:, :, :] = laminar_densities[None, nmin:, None]

    else:

        raise NotImplementedError

    # _validate(dgrid, laminar_densities, nmin, N_ygrid)

    return VoxelData(dgrid, voxel_dimensions, offset=bounding_box.offset)


def create_uniform_density(intensity, x_range, y_range, z_range, n_x, n_y, n_z):
    """ Creates a uniform voxcel density of given intensity
    """
    e_x, e_y, e_z = extent(x_range, y_range, z_range)

    d_x = e_x / float(n_x)
    d_y = e_y / float(n_y)
    d_z = e_z / float(n_z)

    dgrid = np.zeros((n_x, n_y, n_z), dtype=np.float)

    dgrid[:] = intensity

    return VoxelData(dgrid, (d_x, d_y, d_z), offset=(x_range[0], y_range[0], z_range[0]))
