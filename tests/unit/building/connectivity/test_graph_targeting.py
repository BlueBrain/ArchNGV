import numpy as np
import pytest
from numpy.testing import assert_allclose

from archngv.building.connectivity.detail.gliovascular_generation import graph_targeting


def parametric_line(start, u_dir, t):
    return start + t * u_dir


def line_lengths(points, edges):
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


def sequential_edges(Npoints):
    return np.asarray([(n, n + 1) for n in range(Npoints - 1)], dtype=np.int32)


def _create_test_line(Npoints, dP, target_linear_density):
    l_T = target_linear_density

    start = np.array([1.0, 2.0, 3.0])
    direction = np.array([12.0, -21.0, 100.0])

    u_d = direction / np.linalg.norm(direction)

    points = np.asarray(
        [parametric_line(start, u_d, t) for t in dP * np.arange(0.0, float(Npoints))],
        dtype=np.float32,
    )

    edges = sequential_edges(Npoints)

    Ntargets = int(np.round(line_lengths(points, edges).sum() * l_T))

    targets = np.asarray(
        [parametric_line(start, u_d, t) for t in (1.0 / l_T) * np.arange(0.0, float(Ntargets))],
        dtype=np.float32,
    )

    return points, edges, targets


def test_targeting_on_straight_line():
    def _format_output(a_targets, e_targets):
        return (
            "\n Mismatch in target point generation " "\n\n Actual: \n{0},\n\n Expected: \n{1}"
        ).format(a_targets, e_targets)

    linear_density = 1.0 / 4.7

    points, edges, e_targets = _create_test_line(100, 2.3, linear_density)

    a_targets, a_segments = graph_targeting._distribution_on_line_graph(
        points[edges[:, 0]], points[edges[:, 1]], linear_density
    )

    assert np.allclose(a_targets, e_targets), _format_output(a_targets, e_targets)


@pytest.mark.parametrize(
    "step_size, linear_density",
    [
        (3.61816436, 0.02763832),
        (31.05553495, 0.00322004),
        (18.86324392, 0.00530132),
        (7.67710489, 0.01302574),
        (94.63568747, 0.00105668),
        (26.16099122, 0.00382249),
        (96.88912477, 0.00103211),
        (13.84102589, 0.0072249),
        (99.9146877, 0.00100085),
        (70.11148696, 0.0014263),
    ],
)
def test_targeting_on_random_lines(step_size, linear_density):
    points, edges, e_targets = _create_test_line(20, step_size, linear_density)

    a_targets, a_segments = graph_targeting._distribution_on_line_graph(
        points[edges[:, 0]], points[edges[:, 1]], linear_density
    )

    assert_allclose(a_targets, e_targets)


def test_create_targets():
    points = np.array(
        [
            [0, 0, 0],
            [0, 0, 10],
            [0, 0, 19],
            [0, 0, 21],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )
    positions, segments = graph_targeting.create_targets(points, edges, {"linear_density": 0.25})
    assert_allclose(
        positions,
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0], [0.0, 0.0, 12.0], [0.0, 0.0, 16.0]]
        ),
    )
    assert_allclose(segments, np.array([0, 0, 0, 1, 1], dtype=np.uint64))
