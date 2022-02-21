from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from archngv.building.microdomain.overlap import convex_polygon_with_overlap
from archngv.core.datasets import Microdomain

"""
def test_convex_polygon_with_overlap():

    overlap = np.random.uniform(0., 1.)

    points = 100. + np.random.random((100, 3))

    cp1 = Microdomain.from_convex_hull(ConvexHull(points))
    cp2 = deepcopy(cp1)

    cp2.points = convex_polygon_with_overlap(cp1.centroid, cp1.points, overlap)

    V1 = cp1.volume
    V2 = cp2.volume

    res_overlap = (V2 - V1) / V1

    assert np.isclose(res_overlap, overlap)
"""
