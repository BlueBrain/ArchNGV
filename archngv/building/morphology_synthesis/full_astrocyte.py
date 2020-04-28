""" Synthesis entry function """

import os
import logging
from collections import namedtuple

import numpy as np

import morphio
from tns import AstrocyteGrower
from tns.spatial.point_cloud import PointCloud  # pylint: disable=import-error

from archngv.utils.decorators import log_execution_time, log_start_end
from archngv.exceptions import NGVError

from archngv.building.morphology_synthesis.perimeters import add_perimeters_to_morphology

from .data_extraction import obtain_endfeet_data
from .data_extraction import obtain_synapse_data
from .data_extraction import obtain_cell_properties

from .tns_wrapper import create_tns_inputs


L = logging.getLogger(__name__)


SynthesisInputPaths = namedtuple('SynthesisInputPaths', [
    'cell_data',
    'microdomains',
    'synaptic_data',
    'gliovascular_data',
    'gliovascular_connectivity',
    'neuroglial_connectivity',
    'tns_parameters',
    'tns_distributions',
    'morphology_directory',
    'endfeet_areas'])


@log_start_end
@log_execution_time
def synthesize_astrocyte(astrocyte_index, paths, parameters):
    """ Synthesize a circuit astrocyte

    Args:
        astrocyte_index: int
            The id of the astrocyte
        paths: SynthesisInputPaths
            The various paths need by this function
        parameters: dict

    """
    properties = obtain_cell_properties(astrocyte_index, paths.cell_data, paths.microdomains)

    synapses = obtain_synapse_data(
        astrocyte_index, paths.synaptic_data, paths.neuroglial_connectivity)

    if synapses is not None:
        # space colonization point cloud parameters
        radius_of_influence, removal_radius = parameters['point_cloud']

        # create point cloud
        point_cloud = PointCloud(synapses.values, radius_of_influence, removal_radius)
    else:
        point_cloud = None

    endfeet_data = obtain_endfeet_data(
        astrocyte_index, paths.gliovascular_data, paths.gliovascular_connectivity, paths.endfeet_areas)

    if endfeet_data is None:
        field_function = None
    else:
        # lambda function is passed from config as string
        lambda_string = parameters['attraction_field']

        # add the targets to grow to and the attraction field
        # function which depends on the distance to the target
        # TODO: consider using Equation instead of `eval`?
        field_function = eval(lambda_string)  # pylint: disable=eval-used

    tns_parameters, tns_distributions, tns_context = create_tns_inputs(
        default_parameters_path=paths.tns_parameters,
        default_distributions_path=paths.tns_distributions,
        soma_position=properties.soma_position,
        soma_radius=properties.soma_radius,
        microdomain=properties.microdomain,
        point_cloud=point_cloud,
        endfeet_data=endfeet_data,
        field_function=field_function,
        endfeet_barcode_scaling=True
    )

    morphology = AstrocyteGrower(
        input_parameters=tns_parameters,
        input_distributions=tns_distributions,
        context=tns_context).grow()

    if parameters['perimeter_distribution']['enabled']:
        L.info('Distributing perimeters...')
        add_perimeters_to_morphology(morphology, parameters['perimeter_distribution'])

    filepath = os.path.join(paths.morphology_directory, properties.name + '.h5')

    # TODO: replace this when direct NEURON ordering write is available in MorphIO
    _write_with_NEURON_ordering_hack(morphology, filepath)
    _sanity_checks(filepath)


def _write_with_NEURON_ordering_hack(morphology, filepath):
    """ Writes the morphology, opens it again in NEURON ordering and then
    writes it again to ensure NEURON ordering
    """
    morphology.write(filepath)
    morphology = morphio.Morphology(filepath, options=morphio.Option.nrn_order)
    morphology.as_mutable().write(filepath)


def _sanity_checks(filepath):
    """ Various checks ensuring morphology corectedness
        - existence of duplicate points
        - at least two points in each section
    """

    for section in morphio.Morphology(filepath).iter():

        if section.is_root:
            continue

        points = section.points
        parent_points = section.parent.points

        try:
            assert np.allclose(points[0], parent_points[-1])
        except AssertionError:
            raise NGVError(
                f'Morphology {filepath} is missing duplicate points.\n'
                f'\t Section {section.id}, Points: {points}, Parent last point: {parent_points}'
            )

        try:
            assert len(points) > 1
        except AssertionError:
            raise NGVError(
                f'Morphology {filepath} has one point sections.\n'
                f'\t Section {section.id}, Points: {points}'
            )
