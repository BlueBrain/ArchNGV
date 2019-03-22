import numpy
from scipy.spatial import cKDTree
from voxcell import VoxelData, VoxcellError

from morphmath import rowwise_dot

import logging


L = logging.getLogger(__name__)


def _create_neighborhood(R, dx):

    assert R > 0.5 * dx

    # N intervals and two half intervals on the grid
    # * | * | * | * | *  -> 0.5 * dx + dx * N + 0.5 * dx = R

    N = numpy.ceil(R / dx)

    nx = numpy.arange(-N, N + 1, dtype=numpy.intp)

    return numpy.vstack(numpy.meshgrid(nx, nx, nx)).reshape(3, -1).T


class RadialBasisApproximator(object):

    def __init__(self, voxelized_data, covariance, threshold_radius):

        self._vdata = voxelized_data
        self._neighborhood_idx = _create_neighborhood(threshold_radius,
                                                      self._vdata.voxel_dimensions[0])

        self._inv_C = numpy.linalg.inv(covariance)
        self._factor = 1. / numpy.sqrt((2. * numpy.pi) ** 3. * numpy.linalg.det(covariance))

        L.info('Covariance: {}'.format(covariance))
        L.info('Number of voxels in the neighborhood: {}'.format(len(self._neighborhood_idx)))

    def _kernel(self, vs):

        ds = numpy.dot(vs, self._inv_C)

        ps = ds[:, 0] * vs[:, 0] + ds[:, 1] * vs[:, 1] + ds[:, 2] * vs[:, 2]

        return numpy.exp(- 0.5 * ps) * self._factor

    def _vectorized_estimation(self, points):

        points_indices =  self._vdata.positions_to_indices(points, strict=False)

        valid_mask = ~numpy.any(points_indices == VoxelData.OUT_OF_BOUNDS, axis=1)

        results = numpy.zeros(len(points), dtype=numpy.float)

        if valid_mask.any():


            neighborhood = points_indices[valid_mask][:, numpy.newaxis] + self._neighborhood_idx + 0.5

            n_points, n_neighbors, n_dims = neighborhood.shape

            flat_nn = neighborhood.reshape(n_neighbors * n_points, n_dims)

            centers = self._vdata.indices_to_positions(flat_nn)
            vectors = centers - numpy.repeat(points[valid_mask], n_neighbors, axis=0)

            kernel_values = self._kernel(vectors).reshape(n_points, n_neighbors)

            voxel_values = self._vdata.lookup(centers, outer_value=0.).reshape(n_points, n_neighbors)

            results[valid_mask] = (kernel_values * voxel_values).sum(axis=1) / kernel_values.sum(axis=1)

        return results

    def _single_estimation(self, point):

        try:

            ijk = self._vdata.positions_to_indices(point)
            neighborhood = self._neighborhood_idx + ijk + 0.5

            voxel_centers = self._vdata.indices_to_positions(neighborhood)

            voxel_values = self._vdata.lookup(voxel_centers, outer_value=0.)

            kernel_values = self._kernel(voxel_centers - point)

            return (kernel_values * voxel_values).sum() / kernel_values.sum()

        except VoxcellError:

            return 0.

    def __call__(self, data):
        return self._single_estimation(data) if data.ndim == 1 else \
               self._vectorized_estimation(data)
