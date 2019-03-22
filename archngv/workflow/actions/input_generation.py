from .log_actions import log_start_end, log_elapsed_time 


@log_start_end
@log_elapsed_time
def neuron_synapse_connectivity(ngv_config, run_parallel):
    """ Extract the connectivity between neurons and synapses
    as they are present in Bluepy. It also outputs the coordinates
    of synapses as synaptic_data.
    """
    from ..detail.input_generation.extract_neuron_synapse_maps import create_neuron_synapse_maps
    create_neuron_synapse_maps(ngv_config, run_parallel)


@log_start_end
@log_elapsed_time
def neuronal_somata_geometries(ngv_config, run_parallel):
    """
    Extract the geometry of the neuronal somata
    """
    from ..detail.input_generation.neuronal_somata_geometries import extract_neuronal_geometry
    extract_neuronal_geometry(ngv_config, run_parallel)


@log_start_end
@log_elapsed_time
def neuronal_somata_spatial_index(ngv_config, run_parallel):
    from ..detail.input_generation.spatial_indexing.neuronal_somata import create_neuronal_somata_spatial_index
    create_neuronal_somata_spatial_index(ngv_config, run_parallel)


@log_start_end
@log_elapsed_time
def neuronal_synapses_spatial_index(ngv_config, run_parallel):
    from ..detail.input_generation.spatial_indexing.neuronal_synapses import create_synapses_spatial_index
    create_synapses_spatial_index(ngv_config, run_parallel)


@log_start_end
@log_elapsed_time
def vasculature_spatial_index(ngv_config, run_parallel):
    from ..detail.input_generation.spatial_indexing.vasculature import vasculature_spatial_index
    vasculature_spatial_index(ngv_config, run_parallel)
