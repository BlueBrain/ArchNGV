""" NGV data exporters """

import logging
import h5py
import numpy as np


L = logging.getLogger(__name__)


FLOAT_DTYPE = np.float32
ID_DTYPE = np.int32


def export_cell_placement_data(filepath, cell_ids, cell_names, somata_positions, somata_radii):
    """ Export cell data """
    cell_names = np.asarray(cell_names, dtype=bytes)

    with h5py.File(filepath, 'w') as fd:

        fd.create_dataset('ids', data=cell_ids)

        dt = h5py.special_dtype(vlen=bytes)
        fd.create_dataset('names', data=cell_names, dtype=dt)

        fd.create_dataset('positions', data=somata_positions, dtype=FLOAT_DTYPE)
        fd.create_dataset('radii', data=somata_radii, dtype=FLOAT_DTYPE)


def export_gliovascular_data(filename, endfeet_surface_coordinates, endfeet_graph_coordinates):
    """ Export endfeet gliovascular data """
    with h5py.File(filename, 'w') as fd:
        fd.create_dataset('endfoot_surface_coordinates', data=endfeet_surface_coordinates, dtype=FLOAT_DTYPE)
        fd.create_dataset('endfoot_graph_coordinates', data=endfeet_graph_coordinates, dtype=FLOAT_DTYPE)


def _write_astrocyte_annotations(astrocyte_name, fd, data):
    """ write astrocyte annotations
    """
    astro_group = fd.create_group(astrocyte_name)

    synapses_subgroup = astro_group.create_group('synapses')

    if 'synapses' in data:

        syn_data = data['synapses']

        synapses_subgroup.create_dataset('ids',
            data=syn_data['synapse_ids'], dtype=np.int64)

        synapses_subgroup.create_dataset('morph_section_ids',
            data=syn_data['morph_locations'][0], dtype=ID_DTYPE)

        synapses_subgroup.create_dataset('morph_segment_ids',
            data=syn_data['morph_locations'][1], dtype=ID_DTYPE)

        synapses_subgroup.create_dataset('morph_segment_offsets',
            data=syn_data['morph_locations'][1], dtype=FLOAT_DTYPE)
    else:
        L.info('No synapse data for astrocyte %s', astrocyte_name)

    endfeet_subgroup = astro_group.create_group('endfeet')

    if 'endfeet' in data:

        endf_data = data['endfeet']

        endfeet_subgroup.create_dataset('morph_section_ids',
            data=endf_data['morph_section_ids'], dtype=ID_DTYPE)

        endfeet_subgroup.create_dataset('compartments',
            data=endf_data['compartments'], dtype=FLOAT_DTYPE)
    else:
        L.info('No endfeet data for astrocyte %s', astrocyte_name)


def _write_astrocyte_properties(astrocyte_name, fd, data):
    """ Write data for astrocyte with astrocyte_name """

    astro_group = fd.create_group(astrocyte_name)

    astro_group.create_dataset('section_perimeters',
        data=data['section_perimeters'], dtype=FLOAT_DTYPE)

    er_subgroup = astro_group.create_group('ER')

    if 'ER' in data:

        er_data = data['ER']

        er_subgroup.create_dataset('morph_section_ids',
            data=er_data['morph_section_ids'], dtype=ID_DTYPE)

        er_subgroup.create_dataset('fractions',
            data=er_data['fractions'], dtype=FLOAT_DTYPE)
    else:
        L.info('No ER for astrocyte %s', astrocyte_name)

    mito_subgroup = astro_group.create_group('mitochondria')

    if 'mitochondria' in data:

        mito_data = data['ER']

        mito_subgroup.create_dataset('morph_section_ids',
            data=mito_data['morph_section_ids'], dtype=ID_DTYPE)

        mito_subgroup.create_dataset('fractions',
            data=mito_data['fractions'], dtype=FLOAT_DTYPE)

    else:
        L.info('No mitochondria for astrocyte %s', astrocyte_name)


def export_annotations_and_properties(annotations_filename, properties_filename, data_iterable):
    """ Export annotations and astrocyte specific data
    Notes:
        Data Layout:
            astro0/
                ER/
                    morph_section_ids
                    fractions [diameter_fraction, perimeter_fraction]
                mitochondria/
                    morph_section_ids
                    fractions
            astro1/
                .
                .
                .
    """
    with h5py.File(annotations_filename, 'w') as annotations_fd, \
         h5py.File(properties_filename, 'w') as properties_fd:
        for name, annotations, properties in data_iterable:
            _write_astrocyte_annotations(name, annotations_fd, annotations)
            _write_astrocyte_properties(name, properties_fd, properties)
