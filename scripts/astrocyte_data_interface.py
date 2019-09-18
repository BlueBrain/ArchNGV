"""
Object oriented way to access NGV astrocyte data
"""

import os
from itertools import starmap
from collections import namedtuple
from cached_property import cached_property
import h5py
import numpy as np

import neurom
from archngv.core import data_structures

# process types
PERIVASCULAR = neurom.AXON
PERISYNAPTIC = neurom.BASAL_DENDRITE


# in the future we will need to add the synapse annotations
PerisynapticProcess = namedtuple('PerisynapticProcess', ['root_section'])
PerivascularProcess = namedtuple('PerivascularProcess', ['root_section',
                                                         'endfoot_area_mesh',
                                                         'endfoot_target',
                                                         'endfoot_annotation'])


EndfootAnnotation = namedtuple('Annotations', ['vessel_segment',
                                               'cell_section',
                                               'cell_segment',
                                               'cell_offset'])


SynapseData = namedtuple('Synapses', ['ids',
                                      'coordinates',
                                      'cell_sections',
                                      'cell_segments',
                                      'cell_offsets'])


MicrodomainData = namedtuple('Microdomain', ['geometry', 'neighbors'])


def _perivascular_process(root_section, endfoot_data):
    """ Creates a perivascular process given the root_section from MorphIO
        and the endfoot data that has been generated from the circuit build
    """
    mesh, target, vessel_id, section, segment, offset = endfoot_data
    return PerivascularProcess(
        root_section=root_section,
        endfoot_area_mesh=mesh,
        endfoot_target=target,
        endfoot_annotation=EndfootAnnotation(vessel_id, section, segment, offset)
    )


def _perisynaptic_process(root_section):
    """ Creates a perisynaptic process given the root_section from MorphIO"""
    return PerisynapticProcess(root_section)


class AstrocyteData:
    """ Data container for Astrocyte related information in the NGV circuit """
    def __init__(self, filepath, microdomain, circuit_data, endfeet_data, synapse_data=None):

        self.filepath = filepath
        self.microdomain = microdomain
        self._endfeet_data = endfeet_data
        self.circuit_data = circuit_data
        self.synapse_data = synapse_data

    @cached_property
    def morphology(self):
        """ Returns morphology object """
        return neurom.load_neuron(self.filepath)

    def _filter_roots(self, neurite_type):
        """ Filters the astrocyte processes based on the process_type """
        return filter(lambda s: s.type == neurite_type, self.morphology.neurites)

    @property
    def perivascular_processes(self):
        """ Returns the perivascular processes of the atrocyte """
        roots_with_data = zip(self._filter_roots(PERIVASCULAR), self._endfeet_data)
        return list(starmap(_perivascular_process, roots_with_data))

    @property
    def perisynaptic_processes(self):
        """ Returns the perisynaptic processes of the astrocyte """
        return list(map(_perisynaptic_process, self._filter_roots(PERISYNAPTIC)))


def _zip_endfeet_data(endfeet_ids, vessels_ids, annotation_path, endfeetome):
    """ Returns a generator of the endfeet data per endfoot """

    endfeet_areas = endfeetome.areas
    endfeet_targets = endfeetome.targets.endfoot_surface_coordinates

    with h5py.File(annotation_path, 'r') as f_d:

        e_loc = f_d['endfeet_location']
        for i, (endfoot_id, vessel_id) in enumerate(zip(endfeet_ids, vessels_ids)):
            yield (
                endfeet_areas[endfoot_id],
                endfeet_targets[endfoot_id],
                vessel_id,
                e_loc["section_id"][i],
                e_loc["segment_id"][i],
                e_loc["segment_offset"][i]
            )


def _circuit_data(astrocyte_point_data):
    """ Returns circuit related info in a structured array """
    circuit_soma_position = astrocyte_point_data[:3]
    circuit_radius = astrocyte_point_data[3]

    struct_dtype = np.dtype([('soma_radius', 'f4'), ('soma_position', 'f4', (3,))])
    return np.array((circuit_radius, circuit_soma_position), dtype=struct_dtype)


def _synapse_data(annotation_path, synapses):
    """ Returns the synapses that are related to the current astrocyte """
    with h5py.File(annotation_path, 'r') as f_d:

        synapse_ids = f_d['synapse_id'][:]
        return SynapseData(synapse_ids,
                           synapses.synapse_coordinates(synapse_ids=synapse_ids),
                           f_d['synapse_location']['section_id'][:],
                           f_d['synapse_location']['segment_id'][:],
                           f_d['synapse_location']['segment_offset'][:])


def astrocyte_data(astrocyte_ids, circuit_directory, with_synapses=False):
    """
    This is a slow object oriented way to access the circuit data from the point of
    view of astrocytes.

    Args:
        astrocyte_ids (iterable)
        circuit_directory (string): absolute path to circuit directory (parent of build/ directory)

    Returns:
        A generator of AstrocyteData objects that correspond to astrocyte_ids.
    """
    cfg = data_structures.NGVConfig.from_file(
        os.path.join(circuit_directory, 'build/ngv_config.json'))

    ngv_circuit = data_structures.NGVCircuit(cfg)

    astrocytes = ngv_circuit.data.astrocytes
    microdomains = ngv_circuit.data.microdomains
    endfeetome = ngv_circuit.data.endfeetome
    synapses = ngv_circuit.data.synapses

    gv_conn = ngv_circuit.connectome.gliovascular

    for astrocyte_id in astrocyte_ids:

        name = astrocytes.astrocyte_names[astrocyte_id].decode('utf-8')
        filepath = os.path.join(cfg.morphology_directory, name + '.h5')

        microdomain_data = MicrodomainData(
            geometry=microdomains[astrocyte_id],
            neighbors=microdomains.domain_neighbors(astrocyte_id))

        circuit_data = _circuit_data(astrocytes.astrocyte_point_data[astrocyte_id])

        endfeet_data = list(_zip_endfeet_data(
            gv_conn.astrocyte.to_endfoot(astrocyte_id),
            gv_conn.astrocyte.to_vasculature_segment(astrocyte_id),
            os.path.join(cfg.morphology_directory, name + '_endfeet_annotation.h5'),
            endfeetome))

        if with_synapses:
            synapse_data = _synapse_data(
                os.path.join(cfg.morphology_directory, name + '_synapse_annotation.h5'),
                synapses)
        else:
            synapse_data = None

        yield AstrocyteData(
            filepath,
            microdomain_data,
            circuit_data,
            endfeet_data,
            synapse_data
        )
