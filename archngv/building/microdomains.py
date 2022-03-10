""" Tesselation generation and overlap
"""

import logging
from typing import List

import numpy as np
import tess

from archngv.core.datasets import Microdomain
from archngv.exceptions import NGVError
from archngv.spatial.bounding_box import BoundingBox
from archngv.utils.ngons import polygons_to_triangles

L = logging.getLogger(__name__)


def generate_microdomain_tessellation(
    generator_points: np.ndarray, generator_radii: np.ndarray, bounding_box: BoundingBox
) -> List[Microdomain]:
    """Creates a Laguerre Tesselation out of generator spheres taking into account
    intersections with the bounding box.

    Args:
        generator_points: 3d float array of the sphere centers.
        generator_radii: 1d float array of the sphere radii.
        bounding_box: The enclosing region of interest.

    Returns:
        The convex polygon tessellation corresponding to the cell microdomains.

    Note:
        The domain polygons will be intersected with the bounding box geometry in the boundaries.
    """
    limits = (bounding_box.min_point, bounding_box.max_point)

    try:
        # calculates the tessellations using voro++ library
        tess_cells = tess.Container(generator_points, limits=limits, radii=generator_radii)
    except ValueError:
        # a value error is thrown when the bounding box is smaller or overlapping with a
        # generator point. In that case relax a bit the bounding box taking into account
        # the spherical extent of the somata
        L.warning("Bounding box smaller or overlapping with a generator point.")
        bounding_box = BoundingBox.from_spheres(generator_points, generator_radii) + bounding_box
        limits = (bounding_box.min_point, bounding_box.max_point)
        tess_cells = tess.Container(generator_points, limits=limits, radii=generator_radii)

    # convert tess cells to archngv.spatial shapes
    return list(map(_microdomain_from_tess_cell, tess_cells))


def _microdomain_from_tess_cell(cell: tess.Cell) -> Microdomain:
    """Converts a tess cell into a Microdomain object"""
    points = np.asarray(cell.vertices(), dtype=np.float32)

    # polygon face neighbors
    neighbors = np.asarray(cell.neighbors(), dtype=np.int64)

    triangles, tris_to_polys_map = polygons_to_triangles(points, cell.face_vertices())
    triangle_data = np.column_stack((tris_to_polys_map, triangles))

    return Microdomain(points, triangle_data, neighbors[tris_to_polys_map])


def convert_to_overlappping_tessellation(
    microdomains: List[Microdomain], overlap_distribution
) -> List[Microdomain]:
    """Given an existing tessellation uniformly expand each convex region in order to
    achieve an overlap with the neighbors given by the overlap distribution. Overlap is
    a percentage determined by overlap = (V_new - V_old) / V_old

    Returns a new tessellation with the scaled domains and same connectivity as the input
    one.
    """
    scaling_factors = map(
        _scaling_factor_from_overlap,
        overlap_distribution.rvs(size=len(microdomains)),
    )

    return [
        domain.scale(scaling_factor)
        for domain, scaling_factor in zip(microdomains, scaling_factors)
    ]


def _scaling_factor_from_overlap(overlap_factor: float) -> float:
    """Given the centroid of a convex polygon and its points,
    uniformly dilate in order to expand by the overlap factor. However
    the neighbors inflate as well. Thus, the result overlap between the cell
    and the union of neighbors will be:

    a = (Vinflated - Vdeflated) / Vinflated

    Vinflated = s^3 Vo
    Vdeflated = (2 - s)^3 Vo

    Therefore the scaling factor s can be estimated from the equation:

    s^3 (2 - a) - 6 s^2 + 12 s - 8 = 0

    Which has three roots, one real and two complex.
    """
    if not 0.0 <= overlap_factor < 2.0:
        raise NGVError(f"Overlaps must be in the [0.0, 2.0) range: {overlap_factor}")

    p = [2.0 - overlap_factor, -6.0, 12.0, -8.0]

    r = np.roots(p)

    real_roots = np.real(r[~np.iscomplex(r)])

    assert len(real_roots) > 0, "No real roots found in overlap equation."

    scaling_factor = real_roots[0]

    L.debug(
        "Overlap Factor: %.3f, Scaling Factor: %.3f, Predicted Overlap: %.3f",
        overlap_factor,
        scaling_factor,
        (scaling_factor**3 - (2.0 - scaling_factor) ** 3) / scaling_factor**3,
    )

    return scaling_factor
