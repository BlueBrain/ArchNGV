""" Uniform grid data structure and algorithms
"""


# pylint: disable = too-many-instance-attributes, useless-object-inheritance, too-few-public-methods


import numpy as np


def cartesian(arr1, arr2, arr3):
    """ Returns the unique combinations of the three arrays
    """
    i_x = np.indices((arr1.size, arr2.size, arr3.size), dtype=np.intp)
    i_x = i_x.reshape(3, -1).T

    i_x[:, 0] = arr1[i_x[:, 0]]
    i_x[:, 1] = arr2[i_x[:, 1]]
    i_x[:, 2] = arr3[i_x[:, 2]]

    return i_x


class UniformGrid(object):
    """ Uniform grid data structure
    """
    def __init__(self, x_range, y_range, z_range, cell_size):

        self.cell_size = cell_size
        self.inv_cell_size = 1. / self.cell_size

        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range
        self.zmin, self.zmax = z_range
        self.xyz_min = np.array([self.xmin, self.ymin, self.zmin])

        # extents
        self.e_x = np.diff(x_range)[0]
        self.e_y = np.diff(y_range)[0]
        self.e_z = np.diff(z_range)[0]

        self.inv_dx = 1. / self.cell_size[0]
        self.inv_dy = 1. / self.cell_size[1]
        self.inv_dz = 1. / self.cell_size[2]

        # number of grid cells per dimension
        self.s_x = np.ceil(self.e_x * self.inv_dx).astype(np.uintp)
        self.s_y = np.ceil(self.e_y * self.inv_dy).astype(np.uintp)
        self.s_z = np.ceil(self.e_z * self.inv_dz).astype(np.uintp)

    def voxel_centers(self):
        """ Returns centers of voxels
        """
        # the start point of the voxel cells for a grid at origin
        starts = cartesian(np.arange(self.s_x) * self.cell_size[0],
                           np.arange(self.s_y) * self.cell_size[1],
                           np.arange(self.s_z) * self.cell_size[2]).astype(np.float)

        # add the mid point and translate to xmin, ymin, zmin
        return (starts + 0.5 * self.cell_size) + np.array((self.xmin, self.ymin, self.zmin))


class LaminarCounterGrid(UniformGrid):
    """ Counter grid on xz slices
    """
    def __init__(self, laminar_densities, x_range, y_range, z_range, cell_size):  # pylint: disable = too-many-arguments
        super(LaminarCounterGrid, self).__init__(x_range, y_range, z_range, cell_size)

        # N_cells need to be indexed as i + sx * k + sx * sz * j
        index = np.zeros(self.s_x * self.s_y * self.s_z, dtype=np.uintp)

        # dx * dy * dz
        voxel_volume = cell_size[0] * cell_size[1] * cell_size[2]

        current_n = 0
        for rho_y in laminar_densities:

            # indices in i and k are stored closer than j
            # a slab will map to a contiguous sx * sz number
            # of elements in N_cells
            next_n = int(current_n + self.s_x * self.s_z)

            # print rho_y, voxel_volume, rho_y * voxel_volume
            index[current_n: next_n] = np.ceil(1e-9 * rho_y * voxel_volume).astype(np.uintp)

            current_n = next_n

        self.index = index

    def flat_index(self, point):
        ''' Convert 3d point to 1D index
        '''
        i = int((point[0] - self.xmin) * self.inv_dx)
        j = int((point[1] - self.ymin) * self.inv_dy)
        k = int((point[2] - self.zmin) * self.inv_dz)

        return int(i + self.s_x * k + self.s_x * self.s_z * j)

    def counts_per_laminar_slab(self):
        ''' Return the number of cells for each laminar depth y set of bins
        '''
        current_n = 0

        while current_n < self.s_x * self.s_y * self.s_z:

            next_n = int(current_n + self.s_x * self.s_z)

            yield self.index[current_n: next_n]

            current_n = next_n
