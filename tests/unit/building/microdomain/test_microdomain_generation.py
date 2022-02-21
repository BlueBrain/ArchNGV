import mock
import numpy as np

from archngv.building.microdomain.generation import _microdomain_from_tess_cell
from archngv.utils.ngons import polygons_to_triangles


def test_microdomain_from_tess_cell():

    tess_domain = mock.Mock()

    points = [(0.1, 0.2, 0.3), (0.0, 4.0, 5.0), (0.0, -1.0, 2.0), (3.0, 4.0, -1.0)]
    face_vertices = [[0, 1, 2, 3]]
    neighbors = [-1]

    tess_domain.vertices = mock.Mock(return_value=points)
    tess_domain.neighbors = mock.Mock(return_value=neighbors)
    tess_domain.face_vertices = mock.Mock(return_value=face_vertices)

    microdomain = _microdomain_from_tess_cell(tess_domain)

    triangles, triangles_to_polygon_map = polygons_to_triangles(
        np.asarray(points), np.asarray(face_vertices)
    )

    np.testing.assert_allclose(microdomain.points, points)
    np.testing.assert_allclose(microdomain._triangles, triangles)
    np.testing.assert_allclose(microdomain.neighbor_ids, [-1, -1])

    for poly, exp_poly in zip(microdomain.polygons, face_vertices):
        assert set(poly) == set(exp_poly)
