"""
Atlas related class and functions
"""
import numpy as np


class PlacementVoxelData:
    """Simple composition of voxelized intensity
    and voxelized brain region.

    It is not necessary that these two have the same voxel
    dimensions.
    """

    def __init__(self, voxelized_intensity):
        self.voxelized_intensity = voxelized_intensity
        self._factor = 1.0 / self.voxelized_intensity.voxel_dimensions

    def in_geometry(self, point):
        """Checks if the point is in a valid region by trying to"""
        result = (point - self.voxelized_intensity.offset) * self._factor
        result[np.abs(result) < 1e-7] = 0.0
        return self.voxelized_intensity.raw[int(result[0]), int(result[1]), int(result[2])]
