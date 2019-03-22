
import logging

from .common import H5ContextManager


L = logging.getLogger(__name__)


class GliovascularData(H5ContextManager):

    def __init__(self, filepath):
        super(GliovascularData, self).__init__(filepath)

        self.endfoot_graph_coordinates = \
        self._fd['/endfoot_graph_coordinates']

        self.endfoot_surface_coordinates = \
        self._fd['/endfoot_surface_coordinates']


class GliovascularDataInfo(GliovascularData):

    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('gliovascular_data') 
        super(GliovascularDataInfo, self).__init__(filepath)
        self._config = ngv_config

    def endfoot_mesh_path(self, endfoot_index):
        return os.path.join(self.enfeet_directory, '{}.stl'.format(endfoot_index))

    def endfoot_mesh_object(self, endfoot_index):
        import stl
        return stl.Mesh.from_file(self.endfoot_mesh_path(endfoot_index))
