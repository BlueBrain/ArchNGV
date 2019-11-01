"""
Endfoot equivalent compartment for NEURON
"""

import logging
import numpy as np

from archngv.utils.linear_algebra import principal_directions
from archngv.utils.projections import vectorized_scalar_projection


L = logging.getLogger(__name__)


def _principal_direction(points):
    """ Given an array of points return the principal direction
    of their covariance matrix and the left and right projections
    from the origin along that direction.
    """
    centroid = np.mean(points, axis=0)
    vectors = points - centroid

    # returns sorted dirs by eigenvalue magnitude (big-->small)
    principal_direction = principal_directions(vectors)[0]
    projs = vectorized_scalar_projection(vectors, principal_direction)

    # biggest extend right of zero + biggest extent left of zero
    right_extent, left_extent = max(projs), abs(min(projs))

    return principal_direction, centroid, left_extent, right_extent


def _target_to_maximal_extent(points, target):
    """
    Find the longest vector from the target to the extent points
    along the principal direction of variation of the surface points of
    the endfoot.

           target
         /        |
        /          |
       /            |
      /              |
    p0 -- centroid -- p1 ----> principal direction
       ||          ||
      left       right
      extent     extent

    Returns the direction and length from the target to either p0 or p1.
    """
    p_dir, centroid, left_extent, right_extent = _principal_direction(points)

    p0 = centroid - left_extent * p_dir
    p1 = centroid + right_extent * p_dir

    v0 = p0 - target
    v1 = p1 - target

    l0 = np.linalg.norm(v0)
    l1 = np.linalg.norm(v1)

    if l0 > l1:
        return v0 / l0, l0
    return v1 / l1, l1


def _endfoot_compartment_data(endfoot_target,
                              endfoot_mesh_points,
                              endfoot_mesh_area,
                              endfoot_mesh_thickness):
    """ Given the mesh information of the endfoot, it generates the length,
    diameter and perimeter of an equivalent cylinder that will encode this info.

    From the length and the diameter the volume of the endfoot can be calculated.
    From the length and the perimeter the respective area can be calculated.

    The endfoot is not a cylinder, there the diameter and perimeter are both required
    to encode the information of its geometry.

    Args:
        endfoot_target: array[float, (3,)]
        endfoot_mesh_points: array[float, (N, 3)]
        endfoot_mesh_area: float
        endfoot_mesh_thickness: float

    Returns:
        total_length, diameter, perimeter
    """
    _, total_length = _target_to_maximal_extent(endfoot_mesh_points, endfoot_target)

    endfoot_volume = endfoot_mesh_area * endfoot_mesh_thickness

    diameter = 2.0 * np.sqrt(endfoot_volume / (np.pi * total_length))
    perimeter = endfoot_mesh_area / (np.pi * total_length)

    return total_length, diameter, perimeter


def create_endfeet_compartment_data(_, endfeet_data):
    """ Creates the data that is required to construct endfeet compartments in NEURON, using
    the area mesh and target of the endfoot.

    Returns:
        compartment_data: array[float, (N, 3)]
        The total length, diameter and perimeter for each endfoot compartment that will be
        created in NEURON.
    """
    targets, area_meshes = endfeet_data.targets, endfeet_data.area_meshes
    compartment_data = np.zeros((len(targets), 3), dtype=np.float)

    for i, (endfoot_target, area_mesh) in enumerate(zip(targets, area_meshes)):

        if len(area_mesh.triangles) == 0:
            L.info('Endfoot %d has no triangles. Mesh has not been grown.', area_mesh.index)
            continue

        compartment_data[i] = _endfoot_compartment_data(
            endfoot_target,
            area_mesh.points,
            area_mesh.area,
            area_mesh.thickness)

    return compartment_data
