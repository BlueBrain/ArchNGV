from archngv.workflow.actions.log_actions import log_start_end, log_elapsed_time


@log_start_end
@log_elapsed_time
def cell_placement(config, run_parallel):
    from ..detail.building.cell_placement import create_cell_positions
    create_cell_positions(config, run_parallel)


@log_start_end
@log_elapsed_time
def microdomain_tesselation(config, run_parallel):
    from ..detail.building.microdomains import create_microdomains
    create_microdomains(config, run_parallel)


@log_start_end
@log_elapsed_time
def gliovascular_connectivity(config, run_parallel):
    from ..detail.building.gliovascular_connectivity import create_gliovascular_connectivity
    create_gliovascular_connectivity(config, run_parallel)


@log_start_end
@log_elapsed_time
def neuroglial_connectivity(config, run_parallel):
    from ..detail.building.neuroglial_connectivity import create_neuroglial_connectivity
    create_neuroglial_connectivity(config, run_parallel)


@log_start_end
@log_elapsed_time
def morphology_synthesis(config, run_parallel):
    from ..detail.building.synthesis import create_synthesized_morphologies
    create_synthesized_morphologies(config, run_parallel)


@log_start_end
@log_elapsed_time
def astrocyte_endfeet_area_reconstruction(config, run_parallel):
    from ..detail.building.endfeet_area_reconstruction import create_endfeet_areas
    create_endfeet_areas(config, run_parallel)

