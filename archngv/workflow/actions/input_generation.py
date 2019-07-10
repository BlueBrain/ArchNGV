from archngv.workflow.actions.log_actions import log_start_end, log_elapsed_time


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
