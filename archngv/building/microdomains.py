""" Tessellation generation and overlap
"""

import logging
from typing import Iterator

import numpy as np
import tess

from archngv.core.datasets import Microdomain
from archngv.exceptions import NGVError
from archngv.spatial.bounding_box import BoundingBox
from archngv.utils.linear_algebra import normalize_vectors
from archngv.utils.ngons import polygons_to_triangles

L = logging.getLogger(__name__)


def generate_microdomain_tessellation(
    generator_points: np.ndarray, generator_radii: np.ndarray, bounding_box: BoundingBox
) -> Iterator[Microdomain]:
    """Creates a Laguerre Tessellation out of generator spheres taking into account
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

    return map(_microdomain_from_tess_cell, tess_cells)


def _microdomain_from_tess_cell(cell: tess.Cell) -> Microdomain:
    """Converts a tess cell into a Microdomain object"""
    points = np.asarray(cell.vertices(), dtype=np.float32)

    # polygon face neighbors
    neighbors = np.asarray(cell.neighbors(), dtype=np.int64)

    triangles, tris_to_polys_map = polygons_to_triangles(points, cell.face_vertices())
    triangle_data = np.column_stack((tris_to_polys_map, triangles))

    return Microdomain(points, triangle_data, neighbors[tris_to_polys_map])


def scaling_factor_from_overlap(overlap_factor: float) -> float:
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


def scale_microdomains(microdomains, scaling_factors, bounding_box):
    """Scale microdomains according to scaling factors.

    Boundary domains are snapped to the bbox.
    """
    min_point, max_point = bounding_box.ranges

    for domain, scaling_factor in zip(microdomains, scaling_factors):
        new_domain = domain.scale(scaling_factor)

        # snap the boundary points back to the bbox
        new_domain.points = np.clip(new_domain.points, a_min=min_point, a_max=max_point)

        yield new_domain


def limit_microdomains_to_roi(microdomains, astrocyte_soma_pos, region_mask):
    """
    The input microdomains have been creating in the region of interest bounding box.
    This function creates a new Microdomain if its points need to be transformed
     (points that are located outside the region of interest (roi)
     (using the region atlas) inside the roi, using the astrocyte soma position
     and the the distance from the soma center to the closest domain point),
     otherwise it yield the original

    Args:
        microdomains: Microdomains
        astrocyte_soma_pos: numpy array of shape (N, 3) that represents the astrocytes positions.
        region_mask: region of interest.

    Yields:
        a Microdomain.

    """
    for microdomain, soma_pos in zip(microdomains, astrocyte_soma_pos):
        # small trick to ensure that the points on the walls will not be considered as outside
        new_points = np.clip(microdomain.points, a_min=None, a_max=region_mask.bbox[1] - 1e-5)
        vectors = new_points - soma_pos

        radius = np.linalg.norm(vectors, axis=1).min()

        # Create a mask of the points that lie in out of region voxels
        is_out_of_bounds = region_mask.lookup(new_points, outer_value=0) == 0

        if is_out_of_bounds.any():
            new_points[is_out_of_bounds] = _create_points_on_sphere(
                soma_pos, vectors[is_out_of_bounds], radius
            )

            yield Microdomain(
                points=new_points,
                triangle_data=microdomain.triangle_data,
                neighbors=microdomain.neighbor_ids,
            )
        else:
            yield microdomain


def _create_points_on_sphere(origin, vectors, radius):
    directions = normalize_vectors(vectors)
    return origin + directions * radius
