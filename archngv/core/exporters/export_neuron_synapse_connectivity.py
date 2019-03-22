import os
import h5py
import logging
import numpy as np


L = logging.getLogger(__name__)


def create_synaptic_connectivity_fields(fd_conn, total_neurons, total_synapses):
    """ Creates the main h5 group fields for the different point of views
    """
    # Synapse Point of view Group

    synapse_group = fd_conn.create_group('Synapse')

    dset_afferent_neuron = synapse_group.create_dataset('Afferent Neuron',
                                                        shape=(total_synapses,),
                                                        dtype=np.uintp,
                                                        chunks=None)

    # Afferent Neuron Point of view Group

    afferent_neuron_group = fd_conn.create_group('Afferent Neuron')

    dset_afferent_neuron_offsets = \
        afferent_neuron_group.create_dataset('offsets',
                                             shape=(total_neurons + 1,),
                                             dtype='f16', chunks=None)
    dset_afferent_neuron_offsets[0] = 0
    dset_afferent_neuron_offsets.attrs['column_names'] = ['Synapse']

    return dset_afferent_neuron, dset_afferent_neuron_offsets


def export_neuron_synapse_association(ngv_config,
                                      neuron_gids,
                                      connectome):
    """ Stores the values to the datasets
    """

    n_neurons = len(neuron_gids)
    total_synapses = sum(len(connectome.afferent_synapses(gid)) for gid in neuron_gids)

    L.info('{} total synapses. Started extraction.').format(total_synapses)

    with h5py.File(ngv_config.output_paths('synaptic_data'), 'w') as fd_data, \
         h5py.File(ngv_config.output_paths('synaptic_connectivity'), 'w') as fd_conn:

        dset_syn2gid, dset_gid2off = \
            create_synaptic_connectivity_fields(fd_conn, n_neurons, total_synapses)

        dset_syn_pos = \
            fd_data.create_dataset('synapse_coordinates',
                                   (total_synapses, 3), dtype='f16', chunks=None)

        offset = 0
        for neuron_index, gid in enumerate(neuron_gids):

            gs_pairs = connectome.afferent_synapses(gid)
            n_synapses = len(gs_pairs)

            positions = connectome.synapse_positions(gs_pairs, 'pre', 'center').values

            dset_syn_pos[offset: offset + n_synapses] = \
                connectome.synapse_positions(gs_pairs, 'pre', 'center').values

            dset_syn2gid[offset: offset + n_synapses] = \
                neuron_index * np.ones(n_synapses, dtype=np.uintp)

            offset += n_synapses
            dset_gid2off[neuron_index + 1] = offset

            L.info('Processed neuron {}, synapes: {}'.format(neuron_index, n_synapses))
