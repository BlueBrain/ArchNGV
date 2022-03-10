import mock
import pytest
import numpy as np
from numpy import testing as npt

from archngv.exceptions import NGVError
from archngv.spatial.bounding_box import BoundingBox
from archngv.building import microdomains as tested
from archngv.utils.ngons import polygons_to_triangles


@pytest.mark.parametrize(
    "points, radii, bbox",
    [
        [
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([1.0, 1.0]),
            BoundingBox(np.array([-2.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0])),
        ],
        # check that if the bbox is overlapping with the generator points, it is relaxed
        # to take into account the extent of the generator spheres
        [
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([1.0, 1.0]),
            BoundingBox(np.array([-2.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0])),
        ],
    ],
)
def test_generate_microdomain_tessellation(points, radii, bbox):

    domain1, domain2 = tested.generate_microdomain_tessellation(points, radii, bbox)

    npt.assert_allclose(domain1.centroid, points[0])
    npt.assert_allclose(domain2.centroid, points[1])

    npt.assert_allclose(
        domain1.points,
        [
            [-2.0, -1.0, -1.0],
            [0.0, -1.0, -1.0],
            [-2.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [-2.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [-2.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )

    npt.assert_allclose(
        domain2.points,
        [
            [0.0, -1.0, -1.0],
            [2.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            [2.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [2.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
    )

    npt.assert_allclose(domain1.volume, 8.0, atol=1e-3)
    npt.assert_allclose(domain2.volume, 8.0, atol=1e-3)


def test_microdomain_from_tess_cell():

    tess_domain = mock.Mock()

    points = [(0.1, 0.2, 0.3), (0.0, 4.0, 5.0), (0.0, -1.0, 2.0), (3.0, 4.0, -1.0)]
    face_vertices = [[0, 1, 2, 3]]
    neighbors = [-1]

    tess_domain.vertices = mock.Mock(return_value=points)
    tess_domain.neighbors = mock.Mock(return_value=neighbors)
    tess_domain.face_vertices = mock.Mock(return_value=face_vertices)

    microdomain = tested._microdomain_from_tess_cell(tess_domain)

    triangles, triangles_to_polygon_map = polygons_to_triangles(
        np.asarray(points), np.asarray(face_vertices)
    )

    npt.assert_allclose(microdomain.points, points)
    npt.assert_allclose(microdomain._triangles, triangles)
    npt.assert_allclose(microdomain.neighbor_ids, [-1, -1])

    for poly, exp_poly in zip(microdomain.polygons, face_vertices):
        assert set(poly) == set(exp_poly)


def test_covert_to_overlapping_tessellation():

    distribution_mock = mock.Mock()
    distribution_mock.rvs.return_value = [1.0, 1.0, 1.0]

    domains = [mock.Mock(scale=lambda v: v) for _ in range(3)]

    overlapping_microdomains = tested.convert_to_overlappping_tessellation(
        domains, distribution_mock
    )

    npt.assert_allclose(overlapping_microdomains, 2.0, atol=1e-3)


def test_scaling_factor_from_overlap():

    npt.assert_almost_equal(
        tested._scaling_factor_from_overlap(0.0),
        1.0,
        decimal=3,
    )

    npt.assert_almost_equal(
        tested._scaling_factor_from_overlap(1.0),
        2.0,
        decimal=3,
    )

    with pytest.raises(NGVError):
        tested._scaling_factor_from_overlap(-1.0)

    with pytest.raises(NGVError):
        tested._scaling_factor_from_overlap(2.1)
