import mock
import numpy as np
import pytest
from numpy import testing as npt
from voxcell import VoxelData

from archngv.building import microdomains as tested
from archngv.core.datasets import Microdomain
from archngv.exceptions import NGVError
from archngv.spatial.bounding_box import BoundingBox
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


def test_scaling_factor_from_overlap():
    npt.assert_almost_equal(
        tested.scaling_factor_from_overlap(0.0),
        1.0,
        decimal=3,
    )

    npt.assert_almost_equal(
        tested.scaling_factor_from_overlap(1.0),
        2.0,
        decimal=3,
    )

    with pytest.raises(NGVError):
        tested.scaling_factor_from_overlap(-1.0)

    with pytest.raises(NGVError):
        tested.scaling_factor_from_overlap(2.1)


def test_limit_microdomains_to_roi():
    """
     Test a microdomnain that contains:
      one point outside the roi in the region mask -> position must change (pt1)
      one point outside the the region mask -> position must change (pt2)
      two points inside the ROI -> position must not change (pt0 and pt3)

    Microdomain points
     -1 |pt2|   |   |   |   |
     0  |pt0|   |   |   |   |
     1  |   |   |   |   |   |
     2  |   |   |   |   |   |
     3  |   |   |   |   |   |
     4  |pt1|   |pt3|   |   |
          0   1   2   3   4

    Region Mask
     0  | X | X | X | X | X |
     1  | X | X | X | X | X |
     2  | X | X | X | X | X |
     3  | X | X | X | X | X |
     4  |   | X | X | X | X |
          0   1   2   3   4
    """

    points = np.array([(0.1, 0.2, 0.3), (0.0, 4.0, 0.0), (0.0, -1.0, 0.2), (2.0, 4.0, 0.3)])
    face_vertices = np.array([[0, 1, 2, 3]])
    neighbors = np.array([-1])
    domain = Microdomain(points=points, triangle_data=face_vertices, neighbors=neighbors)
    microdomains = [domain]

    region_mask_raw = np.ones((5, 5, 5), dtype=bool)
    region_mask_raw[0][4][0] = False

    region_mask = VoxelData(region_mask_raw, voxel_dimensions=[1, 1, 1])

    astro_points = [
        np.array([2.0, 2.0, 0.3])
    ]  # => the radius for project_point_to_sphere will be 2.
    # as the the 4th point with [2.,4., .3] is the closest point to astro_point [2,2,.3]

    new_microdomains = list(
        tested.limit_microdomains_to_roi(microdomains, astro_points, region_mask)
    )
    npt.assert_array_equal(new_microdomains[0].points[0], [0.1, 0.2, 0.3])
    npt.assert_array_equal(new_microdomains[0].points[3], [2.0, 4.0, 0.3])
    vectors = new_microdomains[0].points - astro_points

    radii = np.linalg.norm(vectors, axis=1)

    npt.assert_almost_equal(radii[1], 2.0)
    npt.assert_almost_equal(radii[2], 2.0)
