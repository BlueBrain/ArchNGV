"""This module combines all the NGV lazy data structures."""


class Atlas:
    """Allows lazy evaluation of the Atlases."""

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath

    def get_atlas(self):
        """Access the actual atlas."""
        from voxcell.voxel_data import VoxelData

        return VoxelData.load_nrrd(self.filepath)
