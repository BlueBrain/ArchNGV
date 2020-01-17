""" NGVData combines all the NGV data structures
"""
from cached_property import cached_property

from voxcell import VoxelData

from archngv.core.data_cells import CellDataInfo
from archngv.core.data_synaptic import SynapticData
from archngv.core.data_endfeet_areas import EndfeetAreas
from archngv.core.data_gliovascular import GliovascularData
from archngv.core.data_microdomains import MicrodomainTesselation
from archngv.core.vasculature_morphology.vasculature import Vasculature


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
    def neurons(self):
        """ Returns neurons population """
        from bluepy.v2 import Circuit
        path = self._config.input_paths('microcircuit_path')
        return Circuit(path + '/CircuitConfig').cells

    @cached_property
    def synapses(self):
        """ Returns synaptic data """
        path = self._config.input_paths('synaptic_data')
        return SynapticData(path)

    @cached_property
    def vasculature(self):
        """ Returns vasculature object """
        return Vasculature.load(self._config.input_paths('vasculature'))

    @cached_property
    def vasculature_mesh(self):
        """ Returns vasculature mesh object """
        import trimesh
        return trimesh.load(self._config.input_paths('vasculature_mesh'))

    @cached_property
    def microdomains(self):
        """ Returns microdomain tesselation """
        path = self._config.output_paths('microdomains')
        return MicrodomainTesselation(path)

    @cached_property
    def overlapping_microdomains(self):
        """ Returns overlapping microdomains """
        path = self._config.output_paths('overlapping_microdomains')
        return MicrodomainTesselation(path)

    @cached_property
    def endfeetome(self):
        """ Returns endfeet areas data """
        return Endfeetome(self._config)

    @cached_property
    def voxelized_intensity(self):
        """ Returns atlas voxelized intensity """
        path = self._config.input_paths('voxelized_intensity')
        return VoxelData.load_nrrd(path)


class Endfeetome:
    """ Endfeetome data for both endfeet areas and target points
    """
    def __init__(self, ngv_config):
        self._config = ngv_config

    @cached_property
    def areas(self):
        """ Get Endfeet Areas """
        path = self._config.output_paths('endfeet_areas')
        return EndfeetAreas(path)

    @cached_property
    def targets(self):
        """ Get Endfeet Targets """
        path = self._config.output_paths('gliovascular_data')
        return GliovascularData(path)
