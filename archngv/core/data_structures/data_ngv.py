""" NGVData combines all the NGV data structures
"""
from cached_property import cached_property

from .data_cells import CellDataInfo
from .data_synaptic import SynapticDataInfo
from .data_spatial_index import SpatialIndexInfo
from .data_gliovascular import GliovascularDataInfo
from .data_microdomains import MicrodomainTesselationInfo
from .data_endfeetome import Endfeetome

from ..vasculature_morphology import Vasculature


class NGVData(object):
    """ Composition of all the ngv rich data structures
    """
    def __init__(self, ngv_config):
        self._config = ngv_config

    @cached_property
    def astrocytes(self):
        """ Returns cell data """
        return CellDataInfo(self._config)

    @cached_property
    def synapses(self):
        """ Returns synaptic data """
        return SynapticDataInfo(self._config)

    @cached_property
    def vasculature(self):
        """ Returns vasculature object """
        return Vasculature.load(self._config.input_paths('vasculature'))

    @cached_property
    def vasculature_mesh(self):
        """ Returns vasculature mesh object """
        import trimesh
        return trimesh.load(self.config.input_paths('vasculature_mesh'))

    @cached_property
    def microdomains(self):
        """ Returns microdomain tesselation """
        return MicrodomainTesselationInfo(self._config)

    @cached_property
    def endfeetome(self):
        """ Returns endfeetome data """
        path = self._config.output_paths('endfeetome')
        return Endfeetome(path)

    @cached_property
    def voxelized_intensity(self):
        path = self.config.input_paths('voxelized_intensity')
        return VoxelData.load_nrrd(path)

    def __enter__(self):
        """ Composition context manager """
        self.synapses.__enter__()
        self.cells.__enter__()
        #self.neuroglial.__enter__()
        self.microdomains.__enter__()
        self.gliovascular.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Called when with goes out of scope """
        self.close()

    def close(self):
        """ Composition context manager close """
        self.synapses.close()
        self.cells.close()
        #self.neuroglial.close()
        self.microdomains.close()
        self.gliovascular.close()
