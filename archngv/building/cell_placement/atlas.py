"""
Atlas related class and functions
"""
import numpy as np


class PlacementVoxelData:
    """ Simple composition of voxelized intensity
    and voxelized brain region.

    It is not necessary that these two have the same voxel
    dimensions.
    """
    def __init__(self, voxelized_intensity, voxelized_regions):
        self.voxelized_intensity = voxelized_intensity
        self.voxelized_regions = voxelized_regions
        self._factor = 1. / self.voxelized_intensity.voxel_dimensions
        self._rshape = self.voxelized_intensity.raw.shape

    def in_geometry(self, point):
        """ Checks if the point is in a valid region by trying to
        """
        result = (point - self.voxelized_intensity.offset) * self._factor
        # rounding errors
        result[np.abs(result) < 1e-7] = 0.

        is_outside_boundaries = \
               int(result[0]) < 0 or \
               int(result[1]) < 0 or \
               int(result[2]) < 0 or \
               int(result[0]) >= self._rshape[0] or \
               int(result[1]) >= self._rshape[1] or \
               int(result[2]) >= self._rshape[2]

        return not is_outside_boundaries
