import pytest
import numpy as np

from archngv.spatial import ConvexPolygon
from archngv.building.morphology_synthesis.domain_boundary import StopAtConvexBoundary
from archngv.app.utils import random_generator


HAZARD_RATE = 0.01


@pytest.fixture
def convex_polygon():
    """ A real life domain """
    points = np.array([[-135.81935278,  634.8777047,   338.16458719],
                         [ -93.35100939,  634.8777047,   398.88488127],
                         [-135.81935278,  637.80281964,  343.68587687],
                         [ -50.99345431,  643.70363649,  337.02070683],
                         [-135.81935278,  634.8777047,   350.25115932],
                         [ -83.80092951,  661.90131722,  378.25529382],
                         [ -95.24677112,  644.7095765,   397.33367781],
                         [-115.0330315,   634.8777047,   385.67822932],
                         [ -88.08243399,  634.8777047,   328.14557669],
                         [ -42.21297597,  634.8777047,   342.18542843],
                         [ -41.09854123,  634.8777047,   338.42435016],
                         [ -41.48688437,  637.91957532,  338.60170204],
                         [ -45.34261913,  646.59918415,  341.30019129],
                         [ -42.21567796,  641.04743702,  339.89599549]])

    triangles = np.array([[ 6,  7,  1],
                           [10,  9,  1],
                           [ 8, 10,  1],
                           [ 0,  8,  1],
                           [ 4,  0,  1],
                           [ 7,  4,  1],
                           [ 5,  6,  1],
                           [12,  5,  1],
                           [13, 12,  1],
                           [ 9, 13,  1],
                           [ 8,  0,  2],
                           [ 3,  8,  2],
                           [12,  3,  2],
                           [ 5, 12,  2],
                           [ 0,  4,  2],
                           [ 6,  5,  2],
                           [ 7,  6,  2],
                           [ 4,  7,  2],
                           [10,  8,  3],
                           [11, 10,  3],
                           [13, 11,  3],
                           [12, 13,  3],
                           [11, 13,  9],
                           [10, 11,  9]])

    return ConvexPolygon(points, triangles)


@pytest.fixture
def domain_boundary(convex_polygon):
    rng = random_generator(seed=0)
    return StopAtConvexBoundary(
        convex_polygon.points,
        convex_polygon.triangles,
        convex_polygon.face_normals,
        hazard_rate=HAZARD_RATE,
        rng=rng
    )


def test_constructor(domain_boundary, convex_polygon):

    assert np.allclose(domain_boundary.face_points, convex_polygon.face_points)
    assert np.allclose(domain_boundary.face_normals, convex_polygon.face_normals)
    assert np.isclose(domain_boundary.hazard_rate, HAZARD_RATE)


def test_survival(domain_boundary):
    """ It should decrease with distance """
    distances = np.linspace(0.0, 100.0, 11)
    survivals = domain_boundary.survival(distances)
    expected_survivals = np.array(
        [1.        , 0.90483742, 0.81873075, 0.74081822, 0.67032005,
        0.60653066, 0.54881164, 0.4965853 , 0.44932896, 0.40656966,0.36787944])

    assert np.allclose(expected_survivals, survivals)


def test_acceptance_criterion(domain_boundary):
    """ We need to check that the acceptance criterion is exponentially distributed """
    np.random.seed(0)

    values = np.random.uniform(0., 500., 1000)

    # values that hasn't stopped
    accepted = np.fromiter(
        (val for val in values if not domain_boundary.acceptance_criterion(val, 1.0)),
        dtype=np.float32
    )

    # density histogram
    bin_values, bin_edges = \
        np.histogram(accepted, bins=100, density=True)

    bin_centers = bin_edges[0:-1] + 0.5 * (bin_edges[1] - bin_edges[0])

    # cumulative hist
    cum_values = np.cumsum(bin_values * np.diff(bin_edges))

    # expected from cumulative - survival function
    sur_values = 1. - domain_boundary.survival(bin_centers)

    mean_squared_error = np.mean((cum_values - sur_values) ** 2)

    assert mean_squared_error < 0.001


def test__call__(domain_boundary, convex_polygon):
    # does not stop
    assert not domain_boundary(convex_polygon.centroid, 1.0)
    # stops
    assert domain_boundary([1e6, 1e6, 1e6], 1.0)
