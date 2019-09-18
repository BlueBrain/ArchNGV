"""
Endfoot equivalent compartment for NEURON
"""

import logging
import numpy as np
from scipy.spatial import cKDTree

import morphio

from archngv.utils.decorators import log_start_end
from archngv.utils.decorators import log_execution_time
from archngv.utils.linear_algebra import principal_directions
from archngv.utils.projections import vectorized_scalar_projection

from .annotation import _is_endfoot_termination

L = logging.getLogger(__name__)


N_POINTS = 4
FAKE_VALUE = 0.01


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


def _stump_section(start_point, section_direction):
    """ Create a stump section with four points, necessary for having a bifurcation point.
    It has no other practical reason.
    Returns:
        The points, diameters and perimeters of the stump section
    """
    stump_points = np.asarray([start_point + FAKE_VALUE * float(i) * section_direction for i in range(N_POINTS)])
    stump_diameters = stump_perimeters = [FAKE_VALUE] * N_POINTS

    return stump_points.tolist(), stump_diameters, stump_perimeters


def _endfoot_section(start_point,
                     total_area,
                     total_volume,
                     total_length,
                     section_direction):
    """
    Given the total area and a list of segment lengths, encode the total
    area information into the parameters attribute on each node of the connected
    segments (section).

    Note: The endfoot section is always a cylinder, thus:
    Perimeter = Total Area / Section Length
    Diameter = 2 * sqrt( Total Volume / ( pi * Total Length ))

    Args:
        total_area : float
            Total area of the endfoot surface
        segment_lengths: list[float, (N,)]
            The lengths of the segments in the section

    Returns: list[float, (N + 1,)]
        The equivalent perimeters in each node in the section.
    """
    intervals = total_length * np.linspace(0.0, 1.0, N_POINTS)[:, np.newaxis]
    section_points = start_point + intervals * section_direction

    diameter = 2.0 * np.sqrt(total_volume / (np.pi * total_length))
    perimeter = total_area / (np.pi * total_length)

    return section_points.tolist(), [diameter] * N_POINTS, [perimeter] * N_POINTS


def _add_compartments(section,
                      endfoot_points,
                      endfoot_target,
                      endfoot_volume,
                      endfoot_area):
    """
    Append two sections at the termination that connects to the vasculature
    mesh. One child is a stump of same type as its parent and the second is
    the endfoot compartment section which encodes the mesh surface area and
    and its volume in the perimeters and diameters respectively of the section.

    Args:
        endfoot_points: array[float, (N, 3)]
            The coordinates of the endfoot surface mesh on the vasculature
        endfoot_target: array[float, (3,)]
            The coordinates of the termination point of the morphology
        endfoot_volume: float
            The total volume of the endfoot attached on the vasculature
        endfoot_area: float
            The total area of the endfoot attached on the vasculature
    """
    direction, extent = _target_to_maximal_extent(endfoot_points, endfoot_target)

    section_last_point = section.points[-1]
    s_points, s_diameters, s_perimeters = _stump_section(section_last_point, -direction)  # opposite direction

    section.append_section(
        morphio.PointLevel(s_points, s_diameters, s_perimeters),
        morphio.SectionType(section.type)  # same as parent
    )

    # TODO: endfoot should have its own type, not apical!
    s_points, s_diameters, s_perimeters = _endfoot_section(
        section_last_point,
        endfoot_area,
        endfoot_volume,
        extent,
        direction)

    section.append_section(
        morphio.PointLevel(s_points, s_diameters, s_perimeters),
        morphio.SectionType(morphio.SectionType.apical_dendrite)
    )


@log_start_end
@log_execution_time
def add_endfeet_compartments(morphology, endfeet_data):
    """ For each endfoot area a section is added at the corresponding
    perivascular process.
    """
    targets, area_meshes = endfeet_data.targets, endfeet_data.area_meshes

    points, section_ids = [], []
    for section in filter(_is_endfoot_termination, morphology.iter()):
        points.append(section.points[-1])
        section_ids.append(section.id)

    points = np.asarray(points)
    kd_tree = cKDTree(points, copy_data=False)

    visited_sections = set()
    for endfoot_target, area_mesh in zip(targets, area_meshes):

        if len(area_mesh.triangles) < 1:
            L.info('Endfoot with index %d has no triangles.', area_mesh.index)
            continue

        _, point_id = kd_tree.query(endfoot_target)
        section_id = section_ids[point_id]

        section = morphology.section(section_id)

        # ensure it's a termination!
        assert not section.children, (visited_sections, section_id, section_ids)

        visited_sections.add(section_id)
        endfoot_area = area_mesh.area
        endfoot_volume = endfoot_area * area_mesh.thickness

        _add_compartments(section, area_mesh.points, endfoot_target, endfoot_volume, endfoot_area)
