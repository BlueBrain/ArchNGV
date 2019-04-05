"""
Entry fucntions to use cell placement
"""

# pylint: disable = logging-format-interpolation

import logging
import numpy as np

from .energy import EnergyOperator
from .atlas import PlacementVoxelData
from .generation import PlacementGenerator
from .generation import PlacementParameters
from .soma_generation import truncated_normal_distribution


L = logging.getLogger(__name__)


def total_number_of_cells(voxelized_intensity):
    """ Given a 3D intensity array return the total number of cells
    """
    return int(np.round(voxelized_intensity.voxel_volume * voxelized_intensity.raw.sum() * 1e-9))


def create_placement_parameters(user_params):
    """ Create placement parameters named tuple
    """
    return PlacementParameters(
                                beta = user_params['beta'],
                                number_of_trials = user_params['ntrials'],
                                cutoff_radius = user_params['cutoff_radius'],
                                initial_sample_size = user_params['n_initial']
                              )


def create_positions(parameters,
                     voxelized_intensity,
                     voxelized_brain_regions,
                     spatial_indexes=None):
    """ Placement function that generates positions given the parameters, density and spatial
    indexes

    Returns positions, radii for the spheres
    """
    soma_data = parameters['soma_radius']

    spatial_indexes = [] if spatial_indexes is None else spatial_indexes
    L.info('Number of other Indexes: {}'.format(len(spatial_indexes)))

    total_cells = total_number_of_cells(voxelized_intensity)
    L.info('Total number of cells: {}'.format(total_cells))

    energy_operator = EnergyOperator(voxelized_intensity, parameters['Energy'])
    L.info('Energy operator with parameters: {}'.format(parameters['Energy']))

    soma_distribution = truncated_normal_distribution(soma_data)
    L.info('Truncated Normal Soma Distr: mean: {}, std: {}, low: {}, high: {}'.format(*soma_data))

    placement_data = PlacementVoxelData(voxelized_intensity, voxelized_brain_regions)
    L.info('Voxelized Intensity, Brain Regions shapes: {}, {}'.format(voxelized_intensity.raw.shape,
                                                                      voxelized_brain_regions.raw.shape))

    placement_parameters = \
        create_placement_parameters(parameters['MetropolisHastings'])

    pgen = PlacementGenerator(placement_parameters,
                              total_cells,
                              placement_data,
                              energy_operator,
                              spatial_indexes,
                              soma_distribution)

    L.info('Placement Generator Initializes.')

    pgen.run()

    return pgen.pattern.coordinates, pgen.pattern.radii
