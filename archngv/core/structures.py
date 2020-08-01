"""This module combines all the NGV lazy data structures."""
from cached_property import cached_property


class Microdomains:
    """Allows lazy evaluation of the Microdomain."""
    def __init__(self, microdomain_path, overlaping_path):
        self._microdomain_path = microdomain_path
        self._overlaping_path = overlaping_path

    @cached_property
    def tesselation(self):
        """Access the tesselation of the microdomain."""
        from archngv.core.datasets import MicrodomainTesselation
        return MicrodomainTesselation(self._microdomain_path)

    @cached_property
    def overlapping(self):
        """Returns overlapping microdomains."""
        from archngv.core.datasets import MicrodomainTesselation
        return MicrodomainTesselation(self._overlaping_path)


class Atlas:
    """Allows lazy evaluation of the Atlases."""

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath

    def get_atlas(self):
        """Access the actual atlas."""
        from voxcell.voxel_data import VoxelData
        return VoxelData.load_nrrd(self.filepath)
