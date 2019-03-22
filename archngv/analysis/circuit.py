import os
import pickle
import h5py
import logging
import numpy as np
from cached_property import cached_property

from voxcell.core import VoxelData

from archngv import NGVConfig
from archngv.vasculature_morphology import Vasculature
from archngv.cell_morphology.ngv_collection import NGVCollection
from archngv.cell_morphology import Astrocyte
from archngv.cell_morphology import astrocyte_types
from archngv.connectivity import GliovascularConnectivity


L = logging.getLogger(__name__)


class PlacementData(object):

    def __init__(self, config):
        self.config = config

    def parameters(self):
        return self.config.parameters['cell_placement']

    @cached_property
    def cell_collection_data(self):
        return CellCollectionData(self.config)

    @cached_property
    def voxelized_intensity(self):
        path = self.config.input_paths('voxelized_intensity')
        return VoxelData.load_nrrd(path)

    @cached_property
    def voxelized_regions(self):
        path = self.config.input_paths('voxelized_brain_regions')
        return VoxelData.load_nrrd(path)


class CellCollectionData(object):

    def __init__(self, config):
        self.config = config
        self.cell_collection = NGVCollection.load(self.config)

    @property
    def _astrocyte_mask(self):
        return np.asarray(self.cell_collection.properties['mtype'] == 'ASTROCYTE', dtype=np.bool)

    @property
    def astrocyte_names(self):
        return self.cell_collection.properties['names'][self._astrocyte_mask].values

    @property
    def astrocyte_paths(self):

        pwd = self.config.morphology_directory
        return (os.path.join(pwd, morph_name) + '.h5' for morph_name in self.astrocyte_names)

    @cached_property
    def astrocyte_positions(self):
        return self.cell_collection.mtype_positions('ASTROCYTE')

    @cached_property
    def astrocyte_radii(self):

        mask = self.cell_collection._mask_by_property('mtype', 'ASTROCYTE')

        try:

            radii = self.cell_collection.properties[mask]['radius'].values
            L.info("Radii loaded from cell_collection.")
        except KeyError:

            radii =  self.cell_collection.radii_from_morphologies(self.config.morphology_directory, mask=mask)
            L.info("Radii loaded from morphologies.")
        return radii
    @property
    def astrocyte_morphology_objects(self):


        bulk_file = os.path.join(self.config.morphology_directory, 'morphologies.h5')

        if os.path.isfile(bulk_file):

            L.info('morphologies.h5 found.')
            with h5py.File(bulk_file, 'r') as fp:
                for morph_name in self.astrocyte_names:
                    yield Astrocyte.bulk_load(morph_name, fp)

        else:

            L.info('morphologies.h5 was not found. Attempting to load from separate morphs.')
            for morph_path in self.astrocyte_paths:
                yield Astrocyte.load(morph_path)

    @property
    def astrocytes_endfeet(self):
        return (astro.endfeet for astro in self.astrocyte_morphology_objects)

    @cached_property
    def astrocyte_endfeet_areas(self):

        try:

            path = self.config.output_paths('endfeet_areas')

        except KeyError:

            L.info('Endfeet areas path not found. Attempting to load endfeet_areas.pkl')
            path = os.path.join(self.config.experiment_directory, 'endfeet_areas.pkl')

        with open(path, 'r') as fhandler:
            face_idx, areas = pickle.load(fhandler)

        return face_idx, areas

class GliovascularConnectome(object):

    def __init__(self, config):
        self.config = config
        self._file = None

    @property
    def parameters(self):
        self.config.parameters['gliovascular_connectivity']

    @property
    def endfeet_reachout_strategy(self):
        return self.gliovascular_connectivity_parameters['connection']['Reachout']

    @property
    def endfeet_max_threshold(self):
        return self.gliovascular_connectivity_parameters['connection']['max_number_of_endfeet']

    def gliovascular_connectivity(self):
        self._file = h5py.File(self.config.output_paths('gliovascular_connectivity'), 'r')
        return GliovascularConnectivity.from_filestream(self._file)

    @property
    def graph_targeting(self):
        return self.gliovascular_connectivity.graph_targeting

    @property
    def surface_targeting(self):
        return self.gliovascular_connectivity.surface_targeting

    @property
    def per_astrocyte_surface_targets(self):
        pass

    def __del__(self):
        if self._file is not None:
            self._file.close()


class MicrodomainData(object):

    def __init__(self, config):
        self.config = config

    @property
    def tesselation(self):
        from archngv.microdomain import MicrodomainTesselation
        path = self.config.output_paths('microdomain_structure')
        return MicrodomainTesselation.load(path)


class NeuroglialConnectome(object):

    def __init__(self, config):
        self.config = config

    @property
    def domain_synapses(self):
        return h5py.File(ngv_config.output_paths('microdomain_synapses'))

class NGVCircuit(object):

    def __init__(self, ngv_config):
        self.config = ngv_config

    @cached_property
    def geometry(self):
        raise NotImplementedError

    @cached_property
    def vasculature(self):
        path = self.config.input_paths('vasculature')
        return Vasculature.load(path)

    @cached_property
    def vasculature_mesh(self):
        import trimesh
        path = self._cfg.input_paths('vasculature_mesh')
        return trimesh.load(path)

    @cached_property
    def gliovascular_connectome(self):
        return GliovascularConnectome(self.config)

    @cached_property
    def neuroglial_connectome(self):
        return NeuroglialConnectome(self.config)

    @cached_property
    def placement_data(self):
        return PlacementData(self.config)

    @cached_property
    def microdomain_data(self):
        return MicrodomainData(self.config)
