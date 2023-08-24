""" Synthesis entry function """
import logging

import morphio
import numpy as np
from diameter_synthesis import build_diameters
from morph_tool.resampling import resample_linear_density
from morph_tool.transform import translate
from neurots import AstrocyteGrower

from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data, tns_inputs
from archngv.building.morphology_synthesis.endoplasmic_reticulum import (
    add_endoplasmic_reticulum_to_morphology,
)
from archngv.building.morphology_synthesis.perimeters import add_perimeters_to_morphology
from archngv.building.morphology_synthesis.tns_wrapper import create_tns_inputs
from archngv.exceptions import NGVError
from archngv.utils.decorators import log_execution_time

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
        - no unifurcations

    Args:
        morph (morphio.mut.Morphology): The morphology

    Raises:
        NGVError: If any of the tests are not satisfied
    """
    for section in morph.iter():
        if not (section.is_root or np.allclose(section.points[0], section.parent.points[-1])):
            raise NGVError(
                f"Morphology {morph} is missing duplicate points.\n"
                f"\t Section {section.id}, "
                f"Points: {section.points}, "
                f"Parent last point: {section.parent.points}"
            )

        if not len(section.points) > 1:
            raise NGVError(
                f"Morphology {morph} has one point sections.\n"
                f"\t Section {section.id}, Points: {section.points}"
            )

        if len(section.children) == 1:
            raise NGVError(
                f"Morphology {morph} has unifurcations.\n"
                f"\t Section {section.id}, Child: {section.children[0].id}"
            )


def grow_circuit_astrocyte(
    tns_data,
    properties,
    endfeet_attraction_data,
    space_colonization_data,
    random_generator,
):
    """
    Args:
        tns_data (TNSData): namedtuple of tns parameters, distributions and context
        properties (CellProperties): cell specific properties
        endfeet_attraction_data (EndfeetAttractionData):
            namedtuple that contains data related to endfeet generation
        space_colonization_data (SpaceColonizationData):
            namedtuple that contains data concerning the space colonization
        random_generator (numpy.random.Generator): Random generator instance to use

    Returns:
        morphio.mut.Morphology: The generated astrocyte morphology
    """

    def diametrizer_function(cell, neurite_type, model_params, random_generator):
        # external diametrizer function handle
        return build_diameters.build(
            cell,
            [neurite_type],
            model_params,
            tns_data.parameters["diameter_params"],
            random_generator=random_generator,
        )

    tns_data = create_tns_inputs(
        tns_data=tns_data,
        cell_properties=properties,
        attraction_data=endfeet_attraction_data,
        space_colonization_data=space_colonization_data,
        rng=random_generator,
    )

    return AstrocyteGrower(
        input_parameters=tns_data.parameters,
        input_distributions=tns_data.distributions,
        context=tns_data.context,
        external_diametrizer=diametrizer_function,
        rng_or_seed=random_generator,
    ).grow()


@log_execution_time
def synthesize_astrocyte(astrocyte_index, paths, parameters, random_generator):
    """Synthesize a circuit astrocyte and write it to file

    Args:
        astrocyte_index (int): The id of the astrocyte
        paths (SynthesisInputPaths): The various paths need by this function
        parameters (dict): Input synthesis parameters
        random_generator (numpy.random.Generator): Random generator instance

    Returns: tuple:
        -The cell morphology
        -The cell_property
    """
    (
        cell_properties,
        endfeet_attraction_data,
        space_colonization_data,
    ) = astrocyte_circuit_data(astrocyte_index, paths, random_generator)

    morphology = grow_circuit_astrocyte(
        tns_inputs(paths),
        cell_properties,
        endfeet_attraction_data,
        space_colonization_data,
        random_generator,
    )

    # TODO: Use mut.GlialCell everywhere

    if "resampling" in parameters and parameters["resampling"]["enabled"]:
        L.debug("Resampling morphology...")
        morphology = resample_linear_density(
            morphology,
            parameters["resampling"]["linear-density"],
        )

    if parameters["perimeter_distribution"]["enabled"]:
        L.debug("Distributing perimeters...")
        add_perimeters_to_morphology(
            morphology, parameters["perimeter_distribution"], random_generator
        )

    L.debug("Adding endoplasmic reticulum")
    add_endoplasmic_reticulum_to_morphology(morphology, paths.er_data)

    # TODO: replace this when direct NEURON ordering write is available in MorphIO
    morph = _post_growing(morphology, cell_properties.soma_position)
    _sanity_checks(morph)

    return morph, cell_properties
