"""
Data structures required for gliovascular connectivity
"""
import os
from .common import H5ContextManager


class GliovascularData(H5ContextManager):
    """ Provides access to the endfeet contact points

    Attributes:
        endfoot_graph_coordinates: array[float, (N , 3)]
            Astrocytic endfeet connection point on the skeleton
            of the vasculature.
        endfoot_surface_coordinates: array[float, (N, 3)]
            Astrocytic endfeet connection points on the surface
            of the vasculature.
    """
    def __init__(self, filepath):
        super(GliovascularData, self).__init__(filepath)

        self.endfoot_graph_coordinates = \
            self._fd['/endfoot_graph_coordinates']

        self.endfoot_surface_coordinates = \
            self._fd['/endfoot_surface_coordinates']

    @property
    def n_endfeet(self):
        """ Total number of endfeet """
        return len(self.endfoot_graph_coordinates)


class GliovascularDataInfo(GliovascularData):
    """ Rich access to the endfeet contact points
    """
    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('gliovascular_data')
        super(GliovascularDataInfo, self).__init__(filepath)
        self._config = ngv_config

    def endfoot_mesh_path(self, endfoot_index):
        """ Absolute path for the mesh of the endfoot with endfoot_index
        """
        endfeet_dir = self._config.endfeet_directory
        return os.path.join(endfeet_dir, '{}.stl'.format(endfoot_index))

    def endfoot_mesh_object(self, endfoot_index):
        """ stl object of the endfoot mesh
        """
        import stl
        return stl.Mesh.from_file(self.endfoot_mesh_path(endfoot_index))
