""" Tesselation generation and overlap
"""

import logging
from copy import deepcopy

import numpy as np
import tess

from archngv.core.datasets import Microdomain
from archngv.building.microdomain.overlap import convex_polygon_with_overlap

from archngv.utils.ngons import polygons_to_triangles

L = logging.getLogger(__name__)


def _microdomain_from_tess_cell(cell):
    """ Converts a tess cell into a Microdomain object
    """
    points = np.asarray(cell.vertices(), dtype=np.float32)

    # polygon face neighbors
    neighbors = np.asarray(cell.neighbors(), dtype=np.intp)

    triangles, tris_to_polys_map = polygons_to_triangles(points, cell.face_vertices())
    triangle_data = np.column_stack((tris_to_polys_map, triangles))

    return Microdomain(points, triangle_data, neighbors[tris_to_polys_map])


def generate_microdomain_tesselation(generator_points, generator_radii, bounding_box):
    """ Creates a Laguerre Tesselation out of generator spheres taking into account
    intersections with the bounding box
    """
    limits = (bounding_box.min_point, bounding_box.max_point)

    # calculates the tesselations using voro++ library
    tess_cells = tess.Container(generator_points, limits=limits, radii=generator_radii)

    # convert tess cells to archngv.spatial shapes
    return list(map(_microdomain_from_tess_cell, tess_cells))


def convert_to_overlappping_tesselation(microdomains, overlap_distribution):
    """ Given an existing tesselation uniformly exapnd each convex region in order to
    achieve an overlap with the neighbors given by the overlap distribution. Overlap is
    a percentage determined by overlap = (V_new - V_old) / V_old

    Returns a new tesselation with the scaled domains and same connectivity as the input
    one.
    """
    overlaps = overlap_distribution.rvs(size=len(microdomains))
    overlapping_microdomains = deepcopy(microdomains)

    for dom_index, dom in enumerate(microdomains):

        new_points = convex_polygon_with_overlap(dom.centroid, dom.points, overlaps[dom_index])
        overlapping_microdomains[dom_index].points = new_points

    return overlapping_microdomains
