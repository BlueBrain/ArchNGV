""" Tesselation generation and overlap
"""

import logging
from copy import deepcopy

import numpy as np
import tess

from morphspatial import ConvexPolygon

from archngv.core.microdomain.tesselation import MicrodomainTesselation
from archngv.core.microdomain.overlap import convex_polygon_with_overlap


L = logging.getLogger(__name__)


def _shape_from_cell(cell):
    """ Returns a shape from a tess cell
    """
    points = np.asarray(cell.vertices())
    face_vertices = np.asarray(cell.face_vertices())
    return ConvexPolygon(points, face_vertices)


def _connectivity_from_cells(cells):
    """ Returns the connectivity between the tesselation cells
    """
    extract_neighbors = \
        lambda cell: tuple(neighbor for neighbor in cell.neighbors() if neighbor >= 0)

    return tuple(extract_neighbors(cell) for cell in cells)


def generate_microdomain_tesselation(generator_points, generator_radii, bounding_box):
    """ Creates a Laguerre Tesselation out of generator spheres taking into account
    intersections with the bounding box
    """
    limits = (bounding_box.min_point, bounding_box.max_point)

    # calculates the tesselations using voro++ library
    cells = tess.Container(generator_points, limits=limits, radii=generator_radii)

    # convert tess cells to morphspatial shapes
    regions = list(map(_shape_from_cell, cells))
    connectivity = _connectivity_from_cells(cells)

    return MicrodomainTesselation(regions, connectivity)


def convert_to_overlappping_tesselation(microdomain_tesselation, overlap_distribution):
    """ Given an existing tesselation uniformly exapnd each convex region in order to
    achieve an overlap with the neighbors given by the overlap distribution. Overlap is
    a percentage determined by overlap = (V_new - V_old) / V_old

    Returns a new tesselation with the scaled domains and same connectivity as the input
    one.
    """
    overlaps = overlap_distribution.rvs(size=len(microdomain_tesselation))

    new_regions = list(map(deepcopy, microdomain_tesselation.regions))

    for index, reg in enumerate(new_regions):
        reg.points = convex_polygon_with_overlap(reg.centroid, reg.points, overlaps[index])

    return microdomain_tesselation.with_regions(new_regions)
