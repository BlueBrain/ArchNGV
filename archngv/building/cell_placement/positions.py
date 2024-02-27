"""
Entry functions to use cell placement
"""

import logging

import numpy as np

from archngv.building.cell_placement.atlas import PlacementVoxelData
from archngv.building.cell_placement.energy import EnergyOperator
from archngv.building.cell_placement.generation import (
    PlacementParameters,
    VoxelPlacementGenerator,
    get_cell_count,
)
from archngv.building.cell_placement.soma_generation import truncated_normal_distribution

L = logging.getLogger(__name__)




def create_placement_parameters(user_params):
    """Create placement parameters named tuple"""
    return PlacementParameters(
        beta=user_params["beta"],
        number_of_trials=user_params["ntrials"],
        cutoff_radius=user_params["cutoff_radius"],
        initial_sample_size=user_params["n_initial"],
    )


def create_positions(parameters, voxelized_intensity, spatial_indexes=None):
    """Placement function that generates positions given the parameters, density and spatial
    indexes

    Returns positions, radii for the spheres
    """
    soma_data = parameters["soma_radius"]

    spatial_indexes = [] if spatial_indexes is None else spatial_indexes
    L.info("Number of other Indexes: %d", len(spatial_indexes))

    cell_count_per_voxel, total_cells = get_cell_count(voxelized_intensity)
    L.info("Total number of cells: %d", total_cells)

    energy_operator = EnergyOperator(voxelized_intensity, parameters["Energy"])
    L.info("Energy operator with parameters: %s", parameters["Energy"])

    soma_distribution = truncated_normal_distribution(soma_data)
    L.info("Truncated Normal Soma Distr: mean: %.3f, std: %.3f, low: %.3f, high: %.3f", *soma_data)
    placement_data = PlacementVoxelData(voxelized_intensity)
    L.info(
        "Voxelized Intensity shape %s",
        voxelized_intensity.raw.shape,
    )

    placement_parameters = create_placement_parameters(parameters["MetropolisHastings"])

    pgen = VoxelPlacementGenerator(
        placement_parameters,
        total_cells,
        placement_data,
        energy_operator,
        spatial_indexes,
        soma_distribution,
    )

    L.info("Placement Generator Initializes.")
    
    pgen.run()

    return pgen.pattern.coordinates, pgen.pattern.radii
