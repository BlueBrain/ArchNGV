"""This module combines all the NGV lazy data structures."""
from cached_property import cached_property


class Vasculature:
    """Allows lazy evaluation of the vasculature objects."""

    def __init__(self, vasculature_path, vasculature_mesh_path):
        self._vasculature_path = vasculature_path
        self._vasculature_mesh_path = vasculature_mesh_path

    @cached_property
    def morphology(self):
        """Returns vasculature object."""
        from archngv.core.datasets import Vasculature as VasculatureMorphology
        return VasculatureMorphology.load(self._vasculature_path)

    @cached_property
    def mesh(self):
        """Returns vasculature mesh object."""
        import trimesh
        return trimesh.load(self._vasculature_mesh_path)


class Endfeetome:
    """Allows lazy evaluation of the Endfeetome objects."""

    def __init__(self, areas_path, data_path):
        self._areas_path = areas_path
        self._data_path = data_path

    @cached_property
    def areas(self):
        """Get Endfeet Areas."""
        from archngv.core.datasets import EndfeetAreas
        return EndfeetAreas(self._areas_path)

    @cached_property
    def targets(self):
        """Get Endfeet Targets."""
        from archngv.core.datasets import GliovascularData
        return GliovascularData(self._data_path)


class Microdomains:
    """Allows lazy evaluation of the Microdomains objects."""

    def __init__(self, microdomain_path, overlaping_path):
        self._microdomain_path = microdomain_path
        self._overlaping_path = overlaping_path

    @cached_property
    def tesselation(self):
        """Returns microdomain tesselation."""
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
        self._name = name
        self._filepath = filepath

    def get_atlas(self):
        """Access the actual atlas."""
        from voxcell.voxel_data import VoxelData
        return VoxelData.load_nrrd(self._filepath)
