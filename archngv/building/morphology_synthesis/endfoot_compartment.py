"""
Endfoot equivalent compartment for NEURON
"""
import logging

import numpy as np

from archngv.utils.projections import vectorized_scalar_projection

L = logging.getLogger(__name__)


def _extent_across_vasculature_segment_medial_axis(points, ref_point, segment):
    """Projects all the vectors from the ref_point to points on the direction of
    the segment and returns the max(proj) - min(proj).
    Args:
        points (np.ndarrat): (N, 3)
        ref_point (np.ndarray): (3,)
        segment (np.ndarray): (2, 3)

    Returns:
        extent (float): max(projections) - min(projections)
    """
    segment_direction = segment[1] - segment[0]
    segment_direction /= np.linalg.norm(segment_direction)
    projs = vectorized_scalar_projection(points - ref_point, segment_direction)
    return np.ptp(projs)


def _endfoot_compartment_features(endfoot_length, endfoot_mesh_area, endfoot_mesh_thickness):
    """Given the mesh information of the endfoot, it generates the length,
    diameter and perimeter of an equivalent cylinder that will encode this info.

    From the length and the diameter the volume of the endfoot can be calculated.
    From the length and the perimeter the respective area can be calculated.

    The endfoot is not a cylinder, there the diameter and perimeter are both required
    to encode the information of its geometry.

    Args:
        endfoot_length (float): Length of compartments
        endfoot_mesh_area (float): Surface area of endfoot
        endfoot_mesh_thickness (float): Thickness of endfoot

    Returns:
        tuple:
            diameter (float): A diameter which can be used to reconstruct the volume
                in combination with the length.
            perimeter (float): A perimeter which can be used to reconstruct the area
                in combination with the lengthj.
    """
    endfoot_volume = endfoot_mesh_area * endfoot_mesh_thickness

    diameter = 2.0 * np.sqrt(endfoot_volume / (np.pi * endfoot_length))
    perimeter = endfoot_mesh_area / endfoot_length

    return diameter, perimeter


def create_endfeet_compartment_data(vasculature_segments, targets, area_meshes):
    """Creates the data that is required to construct endfeet compartments in NEURON, using
    the area mesh and target of the endfoot. The compartment length is calculated by the extent
    of the endfoot across the medial axis of its respective segment. The diameters and perimeters
    correspond to the volumes and area of the endfeet respectively.

    Args:
        vasculature_segments (np.ndarray): (N, 2, 3) Vasculature segments corresponding to
            each endfoot
        targets (np.ndarray): (N, 3) Reference points on the surface of the vasculature
            corresponding to the starting points of the endfeet
        area_meshes (List[namedtuple]): List of namedtuple that have the following endfoot
            mesh fields:
                - index (int): Endfoot Index
                - points (np.ndarray): Mesh points
                - triangles (np.ndarray): Mesh triangles
                - area (float): Endfoot surface area
                - thickness (float): Endfoot thickness

    Returns:
        tuple:
            lengths (np.ndarray): (N,) Compartment lengths
            diameters (np.ndarray): (N,) Compartment diameters
            perimeters (np.ndarray): (N,) Compartment perimeters

    Notes:
        If there are no triangle in and endfoot mesh, it returns [0., 0.,0.]
        If a calculated length is zero, it returns [0., 0., 0.].
    """
    assert len(vasculature_segments) == len(targets) == len(area_meshes), (
        f"Vasculature Segments: {vasculature_segments}\n"
        f"Endfeet targets     : {targets}\n"
        f"Area meshes         : {area_meshes}"
    )

    lengths = np.zeros(len(targets), dtype=np.float32)
    diameters = np.zeros(len(targets), dtype=np.float32)
    perimeters = np.zeros(len(targets), dtype=np.float32)

    for i, (segment, target, mesh) in enumerate(zip(vasculature_segments, targets, area_meshes)):

        if len(mesh.triangles) == 0:
            L.info("Endfoot %d has no triangles. Mesh has not been grown.", mesh.index)
            continue

        lengths[i] = _extent_across_vasculature_segment_medial_axis(mesh.points, target, segment)

        if np.isclose(lengths[i], 0.0):
            L.info("Endfoot %d length is zero.", mesh.index)
            continue

        diameters[i], perimeters[i] = _endfoot_compartment_features(
            lengths[i], mesh.area, mesh.thickness
        )

    return lengths, diameters, perimeters
