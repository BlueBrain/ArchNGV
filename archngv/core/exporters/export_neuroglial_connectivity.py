from builtins import range

import os
import logging

import h5py
import numpy as np


L = logging.getLogger(__name__)


def _part_filename(index):
    return 'ng_{}.h5'.format(index)


def cell_output_path(directory_path, index):
    return os.path.join(directory_path, _part_filename(index))


def export_neuroglial_connectivity(data_iterator,
                                   n_unique_astrocytes,
                                   n_unique_synapses,
                                   n_unique_neurons,
                                   neuroglial_connectivity_filepath):
    """
    Write the connectivity between neurons, synapses and astrocytes to file
    """
    with h5py.File(neuroglial_connectivity_filepath, 'w') as fd:

        astrocyte_group = fd.create_group('/Astrocyte')

        astrocyte_offsets_dset = \
            astrocyte_group.create_dataset('offsets', shape=(n_unique_astrocytes + 1, 2), dtype=np.uintp)

        astrocyte_offsets_dset.attrs['column_names'] = \
            np.array(['synapse', 'neuron'], dtype=h5py.special_dtype(vlen=str))

        astrocyte_synapse_dset = \
            astrocyte_group.create_dataset('synapse', shape=(n_unique_synapses * 2,),
                                            dtype=np.uintp, chunks=(100000,), maxshape=(None,))

        astrocyte_neuron_dset = \
            astrocyte_group.create_dataset('neuron', shape=(n_unique_neurons * 2,),
                                            dtype=np.uintp, chunks=(1000,), maxshape=(None,))

        neuron_astrocytes = [set() for _ in range(n_unique_neurons)]

        synapse_offset = neuron_offset = 0
        for index, (synapse_indices, neuronal_indices) in enumerate(data_iterator):

            N = len(synapse_indices)

            astrocyte_synapse_dset[synapse_offset: synapse_offset + N] = synapse_indices

            synapse_offset += N

            # resize dataset
            if synapse_offset > len(astrocyte_synapse_dset):
                astrocyte_synapse_dset.resize((synapse_offset + 10 * N,))

            ####################################################

            M = len(neuronal_indices)

            # resize dataset
            if neuron_offset + M > len(astrocyte_neuron_dset):
                astrocyte_neuron_dset.resize((neuron_offset + 10 * M,))

            astrocyte_neuron_dset[neuron_offset: neuron_offset + M] = neuronal_indices

            neuron_offset += M

            ####################################################

            astrocyte_offsets_dset[index + 1, 0] = synapse_offset
            astrocyte_offsets_dset[index + 1, 1] = neuron_offset

            for nid in neuronal_indices:
                neuron_astrocytes[int(nid - 1)].add(index)

        if len(astrocyte_synapse_dset) > synapse_offset:
            L.info('Resizing astrocyte_synapse_dset %d -> %d', len(astrocyte_synapse_dset), synapse_offset)
            astrocyte_synapse_dset.resize((synapse_offset,))

        if len(astrocyte_neuron_dset) > neuron_offset:
            L.info('Resizing astrocyte_synapse_dset %d -> %d', len(astrocyte_neuron_dset), neuron_offset)
            astrocyte_neuron_dset.resize((neuron_offset,))

        assert synapse_offset == len(astrocyte_synapse_dset)
        assert neuron_offset == len(astrocyte_neuron_dset)

        #######################################################

        n_astrocytes = sum(len(el) for el in neuron_astrocytes)

        neuron_group = fd.create_group('/Neuron')
        neuron_offsets_dset = neuron_group.create_dataset('offsets', shape=(n_unique_neurons + 1,), dtype=np.uintp)
        neuron_offsets_dset.attrs['column_names'] = np.array(['astrocyte'], dtype=h5py.special_dtype(vlen=str))
        neuron_offsets_dset[0] = 0

        neuron_astrocyte_dset = neuron_group.create_dataset('astrocyte', shape=(n_astrocytes,), dtype=np.uintp)

        offset = 0
        for i, astrocytes in enumerate(neuron_astrocytes):

            N = len(astrocytes)

            neuron_astrocyte_dset[offset: (offset + N)] = sorted(astrocytes)

            offset += N

            neuron_offsets_dset[i + 1] = offset

        L.info('Neuroglial connectivity was written.')


def export_synapse_morphology_association(ngv_config, cell_data):
    """ Export the synapses that each astrocyte morphology incorporates
    """
    morph_dir = ngv_config.morphology_directory

    annotation_suffix = "_synapse_annotation"

    cell_path = lambda cell_name: os.path.join(morph_dir, cell_name + annotation_suffix + '.h5')

    cell_paths = map(cell_path, cell_data.astrocyte_names[:])

    with h5py.File(ngv_config.output_paths('neuroglial_connectivity'), 'r+') as ng_conn:

        astrocyte_group = ng_conn['Astrocyte']

        n_synapses = ng_conn['Astrocyte']['offsets'][-1, 0]

        if 'morphology' in astrocyte_group:
            del astrocyte_group['morphology']

        morphology_dset = astrocyte_group.create_dataset('morphology', shape=(n_synapses, 2), dtype=np.uintp)

        offset = 0

        for cell_path in cell_paths:

            L.info('Extracting location from %s', cell_path)

            with h5py.File(cell_path, 'r') as fd:

                synapse_locations = fd['synapse_location']

                n = len(synapse_locations)

                morphology_dset[offset: offset + n] = synapse_locations

                offset += n
