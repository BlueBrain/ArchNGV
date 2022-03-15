""" Data extraction for synthesis workers
"""
import json
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from archngv.building.morphology_synthesis.data_structures import (
    AstrocyteProperties,
    EndfeetAttractionData,
    EndfeetData,
    SpaceColonizationData,
    TNSData,
)
from archngv.core.datasets import (
    CellData,
    EndfootSurfaceMeshes,
    GliovascularConnectivity,
    Microdomains,
    NeuroglialConnectivity,
    NeuronalConnectivity,
)
from archngv.exceptions import NGVError

L = logging.getLogger(__name__)


TARGET_DENSITY = 1.1


def obtain_endfeet_data(astrocyte_index, gliovascular_connectivity, endfeet_areas_path):
    """Extract the endfeet information from astrocyte_index if any, otherwise return None

    Args:
        astrocyte_index (int): The positional index that represents tha astrocyte entity
        gliovascular_connectivity (str): Path to the gv data file
        endfeet_areas_path (str): Path to the endfeet areas file

    Returns:
        EndfeetData: namedtuple containing endfeet related data
    """
    gliovascular_data = GliovascularConnectivity(gliovascular_connectivity)
    endfeet_indices = gliovascular_data.astrocyte_endfeet(astrocyte_index)

    if len(endfeet_indices) == 0:
        L.warning("No endfeet found for astrocyte index %d", astrocyte_index)
        return None
    targets = gliovascular_data.get_properties(
        ["endfoot_surface_x", "endfoot_surface_y", "endfoot_surface_z"], endfeet_indices
    )

    endfeet_areas = EndfootSurfaceMeshes(endfeet_areas_path)[endfeet_indices]

    L.debug("Found endfeet %s for astrocyte index %d", endfeet_indices, astrocyte_index)
    L.debug("Endfeet Coordinates: %s", targets)
    L.debug("Endfeet Area Meshes: %s", endfeet_areas)

    assert targets.ndim == 2
    return EndfeetData(ids=endfeet_indices, targets=targets, area_meshes=endfeet_areas)


def obtain_cell_properties(astrocyte_index, cell_data_filepath, microdomains_filepath):
    """Extract the cell info (cell_name, pos and radius) and its microdomain
    via the ngv_config and its index.

    Args:
        astrocyte_index (int): The positional index that represents tha astrocyte entity
        cell_data_filepath (str): Path to cell data file
        microdomains_filepath (str): Path to microdomains file

    Returns:
        AstrocyteProperties: namedtuple containing cell related data
    """
    microdomain = Microdomains(microdomains_filepath)[astrocyte_index]

    astrocytes = CellData(cell_data_filepath)
    astrocyte_name = astrocytes.get_properties("morphology", astrocyte_index)[0]
    soma_position = astrocytes.positions(astrocyte_index)[0]
    soma_radius = astrocytes.get_properties("radius", astrocyte_index)[0]

    L.info(
        "Index: %d, Name: %s, Pos: %s, Rad: %s",
        astrocyte_index,
        astrocyte_name,
        soma_position,
        soma_radius,
    )

    return AstrocyteProperties(
        name=astrocyte_name,
        soma_position=soma_position,
        soma_radius=soma_radius,
        microdomain=microdomain,
    )


def obtain_synapse_data(astrocyte_index, synaptic_data_filepath, neuroglial_filepath):
    """Obtain the synapse coordinates that correspond to the microdomain
    of astrocyte_index

    Args:
        astrocyte_index (int): The positional index that represents tha astrocyte entity
        synaptic_data_filepath (str): Path to synaptic data file
        neurogial_conn_filepath (str): Path to neuroglial sonata filepath

    Returns:
        pandas.DataFrame: A dataframe with three columns of the x, y and z coordinates of
        the synapses, and an index representing the synapse id
    """
    synaptic_data = NeuronalConnectivity(synaptic_data_filepath)
    neuroglial_connectivity = NeuroglialConnectivity(neuroglial_filepath)
    synapse_ids = sorted(neuroglial_connectivity.astrocyte_synapses(astrocyte_index))

    if len(synapse_ids) == 0:
        L.warning("No synapses found for astrocyte index %d", astrocyte_index)
        return None
    positions = synaptic_data.synapse_positions(synapse_ids)

    assert positions.ndim == 2
    L.debug("Number of synapses for astro index %d: %d", astrocyte_index, len(positions))
    return pd.DataFrame(index=synapse_ids, data=positions, columns=["x", "y", "z"])


def _create_target_point_cloud(microdomain, synapse_points, target_n_synapses, rng):
    """Uniformly generates points inside the microdomains until the total number of
    synapse point reaches the target_n_synapses. If synapse_points are equal or more
    than target_n_synapses, nothing happens.

    Args:
        microdomain (Microdomain): The bounding region of the astrocyte
        synapse_points (np.ndarray): The synapse points available from the neuronal circuit
        target_n_synapses (int): The desired number of synapses
        rng (RandomState, Generator): Random generator object to use

    Returns:
        np.ndarray: new synapses

    Raises:
        NGVError: If the point cloud cannot be updated with new points
    """
    from archngv.spatial.collision import convex_shape_with_spheres

    n_synapses = len(synapse_points)

    result_points = np.empty((target_n_synapses, 3), dtype=np.float32)
    result_points[:n_synapses] = synapse_points

    xmin, ymin, zmin, xmax, ymax, zmax = microdomain.bounding_box

    total_synapses = n_synapses

    for _ in range(100):

        points = rng.uniform(
            low=(xmin, ymin, zmin), high=(xmax, ymax, zmax), size=(target_n_synapses, 3)
        )

        mask = convex_shape_with_spheres(
            microdomain.face_points,
            microdomain.face_normals,
            points,
            np.zeros(len(points)),
        )

        points = points[mask]
        n_points = len(points)

        if n_points + total_synapses > target_n_synapses:
            result_points[total_synapses::] = points[: (target_n_synapses - total_synapses)]
            break

        result_points[total_synapses : total_synapses + n_points] = points
        total_synapses += n_points

    else:
        raise NGVError(
            "Maximum number of iterations reached."
            "The microdomain geometry cannot be filled with new points"
        )

    return result_points


def _scale_domain(microdomain, scale_factor):
    """Copy domain and scale it"""
    domain = deepcopy(microdomain)
    centroid = domain.centroid
    domain.points = scale_factor * (domain.points - centroid) + centroid
    return domain


def _obtain_point_cloud(
    astrocyte_index,
    microdomains_filepath,
    synaptic_data_filepath,
    neuroglial_conn_filepath,
    target_density,
    rng,
):
    """Given the astrocyte index it returns a point cloud for the astrocyte's microdomains
    and all neighboring ones. If the the density in each microdomain is smaller than the
    target_density, new uniform points inside the domain are created until the target one is
    reached.

    Args:
        astrocyte_index (int): The positional index that represents tha astrocyte entity
        microdomains_filepath (str): Path to microdomains file
        synaptic_data_filepath (str): Path to synaptic data file
        neuroglial_conn_filepath (str): Path to neuroglial connectivity file
        rng (RandomState, Generator): Random generator to use

    Returns:
        np.ndarray: Array of 3D points
    """
    with Microdomains(microdomains_filepath) as microdomains:

        # scale the domain to avoid boundary effects from the point distribution
        # which influences the growing
        microdomain = _scale_domain(microdomains[astrocyte_index], 1.5)

        synapses = obtain_synapse_data(
            astrocyte_index, synaptic_data_filepath, neuroglial_conn_filepath
        )

        if synapses is None:
            return np.empty((0, 3), dtype=np.float32)

        synapse_points = synapses.to_numpy()

        # density : 1.1 synapses / um3
        target_n_synapses = int(np.ceil(target_density * microdomain.volume))

        if target_n_synapses > 1e6:
            L.warning("Attempt to create a high num of synapses: %d", target_n_synapses)
            L.warning("The microdomain must be abnormally big. They will be clamped at 1e6")
            target_n_synapses = int(1e6)

        if len(synapses) < target_n_synapses:
            points = _create_target_point_cloud(
                microdomain, synapses.to_numpy(), target_n_synapses, rng=rng
            )
        else:
            points = synapse_points

        return points


def tns_inputs(paths):
    """Returns the three inputs with all the static info, which does
    not change from astrocyte to astrocyte. Additional info will be later
    added for each astrocyte respectively.

    Args:
        paths (SynthesisInputPaths): Synthesis input paths
    Returns:
        TNSData: namedtuple containing parameters, distributions and context
    """
    with open(paths.tns_parameters, "r") as parameters_fd:
        parameters = json.load(parameters_fd)

    with open(paths.tns_distributions, "r") as distributions_fd:
        distributions = json.load(distributions_fd)

    with open(paths.tns_context, "r") as context_fd:
        context = json.load(context_fd)

    return TNSData(parameters=parameters, distributions=distributions, context=context)


def astrocyte_circuit_data(astrocyte_index, paths, rng):
    """Extract astrocyte circuit information

    Args:
        astrocyte_index (int): Astrocyte positional id
        paths (SynthesisInputPaths): All the input paths to synthesis

    Returns:
        AstrocyteProperties: namedtuple properties
        EndfeetAttractionData: namedtuple with endfeet atraction data
        SpaceColonizationData: namedtuple with space colonization data
        rng: Random generator to use
    """
    properties = obtain_cell_properties(astrocyte_index, paths.astrocytes, paths.microdomains)

    point_cloud = _obtain_point_cloud(
        astrocyte_index,
        paths.microdomains,
        paths.neuronal_connectivity,
        paths.neuroglial_connectivity,
        TARGET_DENSITY,
        rng,
    )

    if point_cloud.size == 0:
        space_colonization_data = None
        L.warning("No point cloud is available for astrocyte %d.", astrocyte_index)
    else:
        space_colonization_data = SpaceColonizationData(point_cloud=point_cloud)

    endfeet_data = obtain_endfeet_data(
        astrocyte_index, paths.gliovascular_connectivity, paths.endfeet_areas
    )

    if endfeet_data is None:
        attraction_data = None
        L.warning("No endfeet for astrocyte %d", astrocyte_index)
    else:
        attraction_data = EndfeetAttractionData(targets=endfeet_data.targets)

    return properties, attraction_data, space_colonization_data
