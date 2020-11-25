""" Synthesis entry function """
import logging
import numpy as np

import morphio
from morph_tool.transform import translate

from tns import AstrocyteGrower
from diameter_synthesis import build_diameters

from archngv.utils.decorators import log_execution_time
from archngv.exceptions import NGVError

from archngv.building.morphology_synthesis.tns_wrapper import create_tns_inputs
from archngv.building.morphology_synthesis.data_extraction import tns_inputs
from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data
from archngv.building.morphology_synthesis.perimeters import add_perimeters_to_morphology

L = logging.getLogger(__name__)


def _post_growing(morphology, position):
    """Returns the same morphology but repositioned and NEURON ordered

    Args:
        morphology (morphio.mut.Morphology): Mutable morphology
        position (numpy.array): the position of the cell somata
    """
    # pylint: disable=no-member
    translate(morphology, -np.asarray(position))
    return morphio.mut.Morphology(morphology, options=morphio.Option.nrn_order)


def _sanity_checks(morph):
    """Various checks ensuring morphology corectedness
        - existence of duplicate points
        - at least two points in each section

    Args:
        morph (morphio.mut.Morphology): The morphology

    Raises:
        NGVError: If any of the tests are not satisfied
    """
    for section in morph.iter():

        if section.is_root:
            continue

        points = section.points
        parent_points = section.parent.points

        if not np.allclose(points[0], parent_points[-1]):
            raise NGVError(
                f'Morphology {morph} is missing duplicate points.\n'
                f'\t Section {section.id}, Points: {points}, Parent last point: {parent_points}'
            )

        if not len(points) > 1:
            raise NGVError(
                f'Morphology {morph} has one point sections.\n'
                f'\t Section {section.id}, Points: {points}'
            )


def grow_circuit_astrocyte(tns_data, properties, endfeet_attraction_data, space_colonization_data):
    """
    Args:
        tns_data (TNSData): namedtuple of tns parameters, distributions and context
        properties (CellProperties): cell specific properties
        endfeet_attraction_data (EndfeetAttractionData):
            namedtuple that contains data related to endfeet generation
        space_colonization_data (SpaceColonizationData):
            namedtuple that contains data concerning the space colonization

    Returns:
        morphio.mut.Morphology: The generated astrocyte morphology
    """
    tns_data = create_tns_inputs(
        tns_data, properties, endfeet_attraction_data, space_colonization_data)
    # external diametrizer function handle
    diametrizer_function = lambda cell, model, neurite_type: build_diameters.build(
        cell, model, [neurite_type], tns_data.parameters['diameter_params'])

    return AstrocyteGrower(
        input_parameters=tns_data.parameters,
        input_distributions=tns_data.distributions,
        context=tns_data.context,
        external_diametrizer=diametrizer_function).grow()


@log_execution_time
def synthesize_astrocyte(astrocyte_index, paths, parameters):
    """ Synthesize a circuit astrocyte and write it to file

    Args:
        astrocyte_index (int): The id of the astrocyte
        paths (SynthesisInputPaths): The various paths need by this function
        parameters (dict): Input synthesis parameters
    """
    (cell_properties, endfeet_attraction_data,
     space_colonization_data) = astrocyte_circuit_data(astrocyte_index, paths)

    morphology = grow_circuit_astrocyte(
        tns_inputs(paths), cell_properties, endfeet_attraction_data, space_colonization_data)

    if parameters['perimeter_distribution']['enabled']:
        L.info('Distributing perimeters...')
        add_perimeters_to_morphology(morphology, parameters['perimeter_distribution'])

    # TODO: replace this when direct NEURON ordering write is available in MorphIO
    morph = _post_growing(morphology, cell_properties.soma_position)
    _sanity_checks(morph)
    return morph
