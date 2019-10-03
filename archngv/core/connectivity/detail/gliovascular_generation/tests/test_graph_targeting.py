from numpy.testing import assert_allclose
import numpy as np

from archngv.core.connectivity.detail.gliovascular_generation import graph_targeting


def parametric_line(start, u_dir, t):
    return start + t * u_dir


def line_lengths(points, edges):
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


def sequential_edges(Npoints):
    return np.asarray([(n, n + 1) for n in range(Npoints - 1)], dtype=np.int)


def _create_test_line(Npoints, dP, target_linear_density):

    l_T = target_linear_density

    start = np.array([1., 2., 3.])
    direction = np.array([12., -21., 100.])

    u_d = direction / np.linalg.norm(direction)

    points = np.asarray([parametric_line(start, u_d, t)
                         for t in dP * np.arange(0., float(Npoints))],
                        dtype=np.float64)

    edges = sequential_edges(Npoints)

    Ntargets = int(np.round(line_lengths(points, edges).sum() * l_T))

    targets = np.asarray([parametric_line(start, u_d, t)
                          for t in (1. / l_T) * np.arange(0., float(Ntargets))],
                         dtype=np.float)

    return points, edges, targets


def test_targeting_on_straight_line():

    def _format_output(a_targets, e_targets):
        return ("\n Mismatch in target point generation "
                "\n\n Actual: \n{0},\n\n Expected: \n{1}").format(a_targets, e_targets)

    linear_density = 1. / 4.7

    points, edges, e_targets = _create_test_line(100, 2.3, linear_density)

    a_targets, a_segments = graph_targeting._distribution_on_line_graph(
        points[edges[:, 0]], points[edges[:, 1]], linear_density)

    assert np.allclose(a_targets, e_targets), _format_output(a_targets, e_targets)


def test_targeting_on_random_lines():

    for _ in range(10):

        L = np.random.uniform(0.02, 100.)
        linear_density = 1. / (10. * np.random.uniform(0.01, L - 0.01))

        points, edges, e_targets = _create_test_line(20, L, linear_density)

        a_targets, a_segments = graph_targeting._distribution_on_line_graph(
            points[edges[:, 0]], points[edges[:, 1]], linear_density)

        txt = "\nRandomized line test Failed: \n"
        txt += "parameters: L = {0}, linear density = {1}\n".format(L, linear_density)
        txt += "Mismatch in target point generation \n\nActual: \n{0},\n\nExpected: \n{1}".format(a_targets, e_targets)

        assert len(a_targets) == len(e_targets), txt
        assert np.allclose(a_targets, e_targets), txt


def test_create_targets():
    points = np.array([[0, 0, 0],
                       [0, 0, 10],
                       [0, 0, 19],
                       [0, 0, 21],
                       ], dtype=np.float64)
    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      ])
    positions, segments = graph_targeting.create_targets(points, edges, {'linear_density': 0.25})
    assert_allclose(positions, np.array([[ 0.,  0.,  0.],
                                         [ 0.,  0.,  4.],
                                         [ 0.,  0.,  8.],
                                         [ 0.,  0., 12.],
                                         [ 0.,  0., 16.]]))
    assert_allclose(segments, np.array([0, 0, 0, 1, 1], dtype=np.uint64))
