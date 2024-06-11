# SPDX-License-Identifier: Apache-2.0

""" Wrapper of TNS AstrocyteGrower. It takes care of all
the functionality required for astrocyte synthesis.
"""
import logging
from copy import deepcopy

import numpy as np
from neurots.morphmath import sample

from archngv.building.morphology_synthesis.data_structures import TNSData
from archngv.building.morphology_synthesis.domain_boundary import StopAtConvexBoundary
from archngv.building.morphology_synthesis.domain_orientation import orientations_from_domain
from archngv.building.types import ASTROCYTE_TO_NEURON

L = logging.getLogger(__name__)


PERISYNAPTIC_TYPE = ASTROCYTE_TO_NEURON["domain_process"]
PERIVASCULAR_TYPE = ASTROCYTE_TO_NEURON["endfoot"]


def create_tns_inputs(
    tns_data,
    cell_properties,
    attraction_data=None,
    space_colonization_data=None,
    rng=np.random,
):
    """Generate inputs for tns astrocyte grower

    Args:
        tns_data (TNSData): namedtuple of tns parameters, distributions and context

        cell_properties (AstrocyteProperties):
            namedtuple of name, soma position and radius and microdomain of astrocyte

        attraction_data (EndfeetAttractionData):
            namedtuple of endfeet targets and attraction field function parameters

        space_colonization_data (SpaceColonizationData):
            namedtuple of point_cloud and space colonization parameters

    Returns:
        TNSData: namedtuple of properties, distributions and context
    """
    parameters, distributions, context = deepcopy(tns_data)

    # set origin and radius of cell soma
    # removed the dtype=np.float32 due to precision loss for some reasons
    parameters["origin"] = np.asarray(cell_properties.soma_position)
    distributions["soma"]["size"] = {"norm": {"mean": cell_properties.soma_radius, "std": 0.0}}

    L.debug("Parameters: %s", parameters)

    microdomain = cell_properties.microdomain

    if microdomain is None:
        L.warning("No microdomain boundary provided.")
    else:
        context["collision_handle"] = StopAtConvexBoundary(
            points=microdomain.points,
            triangles=microdomain.triangles,
            triangle_normals=microdomain.face_normals,
            hazard_rate=0.01,
            rng=rng,
        )

    if space_colonization_data is None:
        point_cloud = np.empty((0, 3), dtype=np.float32)
        L.warning("No point cloud provided.")
    else:
        point_cloud = space_colonization_data.point_cloud

    context["space_colonization"]["point_cloud"] = point_cloud

    if attraction_data is None:
        endfeet_targets = None

        # remove endfeet properties
        parameters["grow_types"].remove(PERIVASCULAR_TYPE)
        parameters["diameter_params"]["neurite_types"].remove(PERIVASCULAR_TYPE)

        L.info("No endfeet available")
    else:
        endfeet_targets = attraction_data.targets
        context["endfeet_targets"] = np.asarray(endfeet_targets, dtype=np.float32)
        parameters[PERIVASCULAR_TYPE]["target_ids"] = np.arange(
            len(endfeet_targets), dtype=np.int32
        )

    # determine the orientations of the primary processes by using
    # the geometry of the microdomain and its respective anisotropy
    (
        parameters[PERISYNAPTIC_TYPE]["orientation"],
        parameters[PERISYNAPTIC_TYPE]["domain_distances"],
    ) = orientations_from_domain(
        cell_properties.soma_position,
        microdomain.points,
        microdomain.triangles,
        sample.n_neurites(distributions[PERISYNAPTIC_TYPE]["num_trees"], random_generator=rng),
        fixed_targets=endfeet_targets,
    )

    for tree_type in parameters["grow_types"]:
        L.debug(
            "Barcode scaling for type %s has been set to %s",
            tree_type,
            parameters[tree_type]["barcode_scaling"],
        )

    return TNSData(parameters, distributions, context)
