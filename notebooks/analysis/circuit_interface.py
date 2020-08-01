import os
from cached_property import cached_property

import numpy as np
from archngv import NGVConfig, NGVCircuit
from bluepysnap import Circuit
from bluepysnap.bbp import Cell


class CircuitInterface:

    def __init__(self, config_path):

        self.config = NGVConfig.from_file(config_path)
        self.circuit = NGVCircuit(self.config)
        self._snap_circuit = Circuit(self._neuronal_circuit_sonata_config)

    @property
    def _neuronal_circuit_sonata_config(self):
        circuit_path = self.config.input_paths('microcircuit_path')
        return os.path.join(circuit_path, 'sonata/circuit_config.json')

    @cached_property
    def vasculature(self):
        return self.circuit.data.vasculature

    @cached_property
    def bbox(self):
        return self.vasculature.bounding_box

    @property
    def neuronal_somata(self):
        import bluepy
        bbox = self.bbox
        path = self.config.input_paths('microcircuit_path')
        points = bluepy.Circuit(path).v2.cells.positions().to_numpy()
        mask = bbox.points_inside(points)
        return points[mask]

    @property
    def microdomains(self):
        return self.circuit.data.microdomains

    @property
    def overlapping_microdomains(self):
        return self.circuit.data.overlapping_microdomains
    
    @cached_property
    def microdomain_volumes(self):
        return np.fromiter((d.volume for d in self.microdomains), dtype=np.float32)

    @cached_property
    def overlapping_microdomain_volumes(self):
        return np.fromiter((d.volume for d in self.overlapping_microdomains), dtype=np.float32)

    @property
    def astrocytes(self):
        return self.circuit.data.astrocytes

    @property
    def n_astrocytes(self):
        return len(self.astrocytes)

    @property
    def astrocyte_filepaths(self):
        ids = self.circuit.data.astrocytes.ids
        return list(map(self.circuit.data.astrocytes.morphology_path, ids))

    @cached_property
    def astrocyte_positions(self):
        return self.astrocytes.positions[:]

    @property
    def astrocyte_radii(self):
        return self.astrocytes.radii[:]

    @property
    def voxel_density(self):
        return self.circuit.data.voxelized_intensity

    def morphology(self, astrocyte_id):
        return self.circuit.data.astrocytes.morphology_object(astrocyte_id)

    def neuronal_morphology(self, neuron_id):
        path = self._snap_circuit.nodes['All'].morph.get_filepath(neuron_id)
        return Morphology(path.replace('.swc', '.h5'))

    @cached_property
    def neuronal_positions(self):
        return self._snap_circuit.nodes['All'].positions().to_numpy(dtype=np.float32)

    @cached_property
    def neuronal_filepaths(self):
        nodes = self._snap_circuit.nodes['All']
        filenames = nodes.get(properties=Cell.MORPHOLOGY).tolist()
        mdir = self._snap_circuit.config['components']['morphologies_dir']
        return [os.path.join(mdir, name + '.h5') for name in filenames]

    @property
    def gv_connectivity(self):
        return self.circuit.connectome.gliovascular

    def endfeet_ids_for_astrocyte(self, astrocyte_id):
        return self.gv_connectivity.astrocyte.to_endfoot(astrocyte_id)
    
    @property
    def endfeetome(self):
        return self.circuit.data.endfeetome

    @property
    def endfeet_meshes(self):
        return self.endfeetome.surface_meshes

    @property
    def endfeet_areas(self):
        return self.endfeet_meshes.get('surface_area')

    @property
    def endfeet_areas_unreduced(self):
        return self.endfeet_meshes.get('unreduced_surface_area')

    @property
    def endfeet_thicknesses(self):
        return self.endfeet_meshes.get('surface_thickness')

    @property
    def endfeet_surface_targets(self):
        return self.endfeetome.targets.vasculature_surface_targets
