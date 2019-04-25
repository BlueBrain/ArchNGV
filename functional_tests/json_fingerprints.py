import json
import numpy as np

from archngv.core.data_structures import NGVData
from archngv.core.data_structures import NGVConfig
from archngv.core.data_structures import NGVConnectome


def create_sample_ids(ngv_data, n_neurons, n_synapses, n_astrocytes, n_endfeet):

    total_neurons = 10000
    total_endfeet = ngv_data.gliovascular.n_endfeet
    total_synapses = len(ngv_data.synapses)
    total_astrocytes = len(ngv_data.cells)

    neuron_ids = \
        sorted(np.random.choice(range(total_neurons), size=n_neurons, replace=False))

    synapse_ids = \
        sorted(np.random.choice(range(total_astrocytes), size=n_synapses, replace=False))

    astrocyte_ids = \
        sorted(np.random.choice(range(total_astrocytes), size=n_astrocytes, replace=False))

    endfoot_ids = \
        sorted(np.random.choice(range(total_endfeet), size=n_endfeet, replace=False))

    return neuron_ids, synapse_ids, astrocyte_ids, endfoot_ids


def create_data_fingerprints(config_path, output_path, n_neurons, n_synapses, n_astrocytes, n_endfeet):

    np.random.seed(0)
    ngv_config = NGVConfig.from_file(config_path)

    json_dict = {}

    with \
        NGVData(ngv_config) as ngv_data, \
        NGVConnectome(ngv_config) as ngv_connectome:

        neuron_ids, synapse_ids, astrocyte_ids, endfoot_ids = \
            create_sample_ids(ngv_data, n_neurons, n_synapses, n_astrocytes, n_endfeet)

        json_dict['ngv_data'] = \
            sample_ngv_data(ngv_data, astrocyte_ids, endfoot_ids, synapse_ids)

        json_dict['ngv_connectome'] = \
            sample_ngv_connectome(ngv_connectome, neuron_ids, astrocyte_ids, endfoot_ids, synapse_ids)

        print(json_dict)
        with open(output_path, 'w') as out_file:
            json.dump(json_dict, out_file, indent=4, ensure_ascii=False)


def sample_ngv_data(ngv_data, astrocyte_ids, endfoot_ids, synapse_ids):

    dict_data = {}

    dict_data['cell_data'] = \
        sample_cell_data(astrocyte_ids, ngv_data.cells)

    dict_data['gliovascular_data'] = \
        sample_gliovascular_data(endfoot_ids, ngv_data.gliovascular)

    dict_data['synaptic_data'] = \
        sample_synaptic_data(synapse_ids, ngv_data.synapses)

    dict_data['microdomain_data'] = \
        sample_microdomain_data(astrocyte_ids, ngv_data.microdomains)

    return dict_data

def sample_ngv_connectome(ngv_connectome, neuron_ids, astrocyte_ids, endfoot_ids, synapse_ids):

    dict_data = {}

    dict_data['gliovascular'] = \
        sample_gliovascular_connectivity(astrocyte_ids, endfoot_ids, ngv_connectome.gliovascular)

    dict_data['neuroglial'] = \
        sample_neuroglial_connectivity(astrocyte_ids, ngv_connectome.neuroglial)

    dict_data['synaptic'] = \
        sample_synaptic_connectivity(synapse_ids, neuron_ids, ngv_connectome.synaptic)

    return dict_data

def sample_cell_data(astrocyte_ids, cell_data):
    """ Update dict_data with cell_data infor for selected indices
    """
    handles = \
    [
        ('astrocyte_positions', cell_data.astrocyte_positions),
        ('astrocyte_radii', cell_data.astrocyte_radii),
        ('astrocyte_gids', cell_data.astrocyte_gids),
    ]

    data = {}
    for key, dset in handles:
        data[key] = \
        {
            'sample': dset[astrocyte_ids].tolist(),
            'size': len(dset),
        }

    return data


def sample_gliovascular_data(endfoot_ids, gv_data):

    handles = \
    [
        ('endfoot_graph_coordinates', gv_data.endfoot_graph_coordinates),
        ('endfoot_surface_coordinates', gv_data.endfoot_surface_coordinates)
    ]

    data = {}
    for key, dset in handles:
        data[key] = \
        {
            'sample': dset[endfoot_ids].astype(np.float32).tolist(),
            'size': len(dset),
        }

    return data


def sample_synaptic_data(synapse_ids, syn_data):

    handles = \
    [
        ('synapse_coordinates', syn_data.synapse_coordinates)
    ]

    data = {}
    for key, dset in handles:
        data[key] = \
        {
            'sample': dset[synapse_ids].astype(np.float32).tolist(),
            'size': len(dset),
        }

    return data


def sample_microdomain_data(astrocyte_ids, mdom_data):

    handles = \
    [
        ('domain_points', mdom_data.domain_points),
        ('domain_triangles', mdom_data.domain_triangles),
        ('domain_neighbors', mdom_data.domain_neighbors),
    ]

    data = {}

    for key, handle in handles:
        data[key] = \
        {
            'sample': [handle(index).tolist() for index in astrocyte_ids]
        }

    return data


def sample_gliovascular_connectivity(astrocyte_ids,
                                     endfoot_ids,
                                     gv_conn):
    data = {}
    astro = gv_conn.astrocyte
    print(astro)
    data['astrocyte'] = \
    {
        'endfeet': [astro.to_endfoot(a_id).tolist() for a_id in astrocyte_ids],
        'vasculature_segments': [astro.to_vasculature_segment(a_id).tolist() for a_id in astrocyte_ids]
    }

    endfoot = gv_conn.endfoot
    data['endfoot'] = \
    {
        'astrocytes': [endfoot.to_astrocyte(e_id).tolist() for e_id in endfoot_ids],
        'vasculature_segments': [endfoot.to_vasculature_segment(e_id).tolist() for e_id in endfoot_ids]
    }

    return data


def sample_neuroglial_connectivity(astrocyte_ids, ng_conn):
    data = {}
    astro = ng_conn.astrocyte
    data['astrocyte'] = \
    {
        'neurons': [astro.to_neuron(a_id).tolist() for a_id in astrocyte_ids],
        'synapses': [astro.to_synapse(a_id).tolist() for a_id in astrocyte_ids]
    }

    return data


def sample_synaptic_connectivity(synapse_ids, neuron_ids, syn_conn):
    data = {}
    synapse = syn_conn.synapse
    data['synapse'] = \
    {
        'neurons': [synapse.to_afferent_neuron(s_id).tolist() for s_id in synapse_ids]
    }

    afferent_neuron = syn_conn.afferent_neuron
    data['afferent_neuron'] = \
    {
        'synapses': [afferent_neuron.to_synapse(n_id).tolist() for n_id in neuron_ids]
    }


    return data



if __name__ == '__main__':
    import sys

    config_path = sys.argv[1]
    output_path = sys.argv[2]
    create_data_fingerprints(config_path, output_path, 10, 10, 10, 10)
