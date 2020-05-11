""" Synthesis entry function """
import os
import logging
import numpy as np

import morphio
from tns import AstrocyteGrower
from diameter_synthesis import build_diameters

from archngv.utils.decorators import log_execution_time
from archngv.exceptions import NGVError

from archngv.building.morphology_synthesis.tns_wrapper import create_tns_inputs
from archngv.building.morphology_synthesis.data_extraction import tns_inputs
from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data
from archngv.building.morphology_synthesis.perimeters import add_perimeters_to_morphology


L = logging.getLogger(__name__)


def _write_with_NEURON_ordering_hack(morphology, filepath):
    """Writes the morphology, opens it again in NEURON ordering and then
    writes it again to ensure NEURON ordering

    Args:
        morphology (morphio.mut.Morphology): Mutable morphology
        filepath (str): Output filepath
    """
    morphology.write(filepath)
    morphology = morphio.Morphology(filepath, options=morphio.Option.nrn_order)
    morphology.as_mutable().write(filepath)


def _sanity_checks(filepath):
    """Various checks ensuring morphology corectedness
        - existence of duplicate points
        - at least two points in each section

    Args:
        filepath (str): The morphology filepath

    Raises:
        NGVError: If any of the tests are not satisfied
    """
    for section in morphio.Morphology(filepath).iter():

        if section.is_root:
            continue

        points = section.points
        parent_points = section.parent.points

        if not np.allclose(points[0], parent_points[-1]):
            raise NGVError(
                f'Morphology {filepath} is missing duplicate points.\n'
                f'\t Section {section.id}, Points: {points}, Parent last point: {parent_points}'
            )

        if not len(points) > 1:
            raise NGVError(
                f'Morphology {filepath} has one point sections.\n'
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
    cell_properties, endfeet_attraction_data, space_colonization_data = astrocyte_circuit_data(
        astrocyte_index, paths, parameters)

    morphology = grow_circuit_astrocyte(
        tns_inputs(paths), cell_properties, endfeet_attraction_data, space_colonization_data)

    if parameters['perimeter_distribution']['enabled']:
        L.info('Distributing perimeters...')
        add_perimeters_to_morphology(morphology, parameters['perimeter_distribution'])

    filepath = os.path.join(paths.morphology_directory, cell_properties.name + '.h5')

    # TODO: replace this when direct NEURON ordering write is available in MorphIO
    _write_with_NEURON_ordering_hack(morphology, filepath)
    _sanity_checks(filepath)
