import pytest
import numpy

from morphspatial import ConvexPolygon

from ...vasculature_morphology import Vasculature
#from ..gliovascular_core import Gliovascular

#from archngv.util.graph.graphs import DirectedGraph


def create_continuous_point_data():

    u = numpy.array([0.2, 0.2, 1.])
    u /= numpy.linalg.norm(u)

    points = u * numpy.arange(-3, 3., 1.)[:, numpy.newaxis]

    radii = numpy.arange(1., len(points) + 1) * 0.01

    edges = numpy.array([(i, i + 1) for i in range(len(points) - 1)])

    return points, edges, radii


class MockVasculature(object):

    def __init__(self):
        self.points, self.edges, self.radii = create_continuous_point_data()

        self.segments_radii = self.radii[self.edges[:, 0]], self.radii[self.edges[:, 1]]

        self.segments = self.points[self.edges[:, 0]], self.points[self.edges[:, 1]]

        self.point_graph = DirectedGraph(self.edges, labels=None)


def _check_arr(arr1, arr2):

    msg = '\nExpected:\n{}\nResult:\n{}'.format(str(arr1), str(arr2))

    assert len(arr1) == len(arr2), msg

    msg = '\nExpected       Result\n'
    msg += '\n'.join(['{}\t{}'.format(a1, a2) for a1, a2 in zip(arr1, arr2)])

    assert numpy.allclose(arr1, arr2), msg


def check_create_potential_targets_on_vasculature(GV, expected):

    graph_targeting = GV.gv_data.graph_targeting

    positions = graph_targeting.positions
    vasculature_segment = graph_targeting.vasculature_segment

    expected_graph_targeting = expected['graph_targeting']
    expected_positions = expected_graph_targeting['positions']
    expected_vasculature_segment = expected_graph_targeting['vasculature_segment']

    _check_arr(expected_positions, positions)
    _check_arr(expected_vasculature_segment, vasculature_segment)


def check_connect_astrocytes_with_vasculature_graph(GV, expected):

    # make sure that the previous data is not changed
    check_create_potential_targets_on_vasculature(GV, expected)

    graph_targeting = GV.gv_data.graph_targeting
    connectivity = graph_targeting.astrocyte_target_edges
    vasculature_segment = graph_targeting.vasculature_segment

    expected_graph_targeting = expected['graph_targeting']
    expected_connectivity = expected_graph_targeting['connectivity']
    expected_vasculature_segment = expected_graph_targeting['vasculature_segment']

    _check_arr(expected_connectivity, connectivity)
    _check_arr(expected_vasculature_segment, vasculature_segment)


def check_map_graph_points_to_vasculature_surface(GV, expected):

    # make sure that the previous data is not changed
    check_create_potential_targets_on_vasculature(GV, expected)
    check_connect_astrocytes_with_vasculature_graph(GV, expected)

    surface_targeting = GV.gv_data.surface_targeting
    positions = surface_targeting.positions
    edges = surface_targeting.astrocyte_target_edges
    vsegs = surface_targeting.vasculature_segment

    expected_surf_targeting = expected['surface_targeting']
    expected_positions = expected_surf_targeting['positions']
    expected_edges = expected_surf_targeting['connectivity']
    expected_vsegs = expected_surf_targeting['vasculature_segment']

    _check_arr(expected_positions, positions)
    _check_arr(expected_edges, edges)
    _check_arr(expected_vsegs, vsegs)


def evaluate_simple_connectivity(convex_shape, vasculature, expected):

    options = {'graph_targeting': {'linear_density': 1.},
               'connection': {'Reachout Strategy': "maximum_reachout",
                              'Endfeet Distribution': (10, 1, 0, 3)},
               'surface_targeting': {}}

    GV = Gliovascular([convex_shape], vasculature, options)
    GV.validate_options()

    GV.create_potential_targets_on_vasculature()
    check_create_potential_targets_on_vasculature(GV, expected)

    GV.connect_astrocytes_with_vasculature_graph()
    check_connect_astrocytes_with_vasculature_graph(GV, expected)

    GV.map_graph_points_to_vasculature_surface()
    check_map_graph_points_to_vasculature_surface(GV, expected)

'''
def test_integration_only_one_in_the_corner():
    """ In this case there is only one point, that of (0,0,0) that
    intersects with the convex shape
    """
    points = numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

    face_vertices = numpy.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = numpy.array([[0., 0., -1.], [0., -1., -0.], [-1., 0., 0.],
                                [0.57735027,  0.57735027,  0.57735027]])

    poly = ConvexPolygon(points, face_vertices)

    assert numpy.allclose(face_normals, poly.face_normals)

    vasc = MockVasculature()

    expected_results = \
    {
        'graph_targeting': {
            'positions'          : vasc.points[:-1],
            'connectivity'       : numpy.array([[0, 3]]),
            'vasculature_segment': numpy.array([0, 1, 2, 3, 4])
        },
        'surface_targeting': {
            'positions'          : numpy.array([[0.03720271, 0.03720271, 0.03720271]]),
            'connectivity'       : numpy.array([[0, 0]]),
            'vasculature_segment': numpy.array([[3]])
        }
    }

    evaluate_simple_connectivity(poly, vasc, expected_results)


def test_integration_two():
    """ In this case there is only one point, that of (0,0,0) that
    intersects with the convex shape
    """
    points = 2. * numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

    face_vertices = numpy.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = numpy.array([[0., 0., -1.], [0., -1., -0.], [-1., 0., 0.],
                                [0.57735027,  0.57735027,  0.57735027]])

    poly = ConvexPolygon(points, face_vertices)

    vasc = MockVasculature()

    expected_results = \
    {
        'graph_targeting': {
            'positions'          : vasc.points[:-1],
            'connectivity'       : numpy.array([[0, 3],
                                                [0, 4]]),
            'vasculature_segment': numpy.array([0, 1, 2, 3, 4])
        },
        'surface_targeting': {
            'positions'          : numpy.array([[0.03720271, 0.03720271, 0.03720271],
                                                [0.16318381, 0.16318381, 1.00057175]]),
            'connectivity'       : numpy.array([[0, 0],
                                                [0, 1]]),
            'vasculature_segment': numpy.array([3, 4])
        }
    }

    evaluate_simple_connectivity(poly, vasc, expected_results)
'''
