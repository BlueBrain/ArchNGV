import numpy as np
import pytest

from archngv.building.morphology_synthesis.domain_orientation import (
    Objective,
    choose_ids,
    orientations_from_domain,
    quadratic_angles,
    remove_overlapping_orientations,
)


def test_overlapping_orientations():
    ref_point = np.array([0.0, 0.0, 0.0])

    points = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 0.0], [1.0, 2.0, 1.0], [3.0, 6.0, 3.0]])

    to_remove = np.array([[4.0, 8.0, 4.0], [0.0, 0.0, 2.0]])

    result_points = remove_overlapping_orientations(ref_point, points, to_remove)

    expected_points = np.array([[2.0, 0.0, 0]])

    assert np.allclose(result_points, expected_points)


class MockObjective:
    def __init__(self, func, points):
        self._f = func
        self.points = points
        self.occupied = np.zeros(len(points), dtype=bool)

    def __call__(self, index, selected):
        return self._f(self.points[index])


def test_choose_ids_1():
    points = np.array([1, 5, 4, 3, 0])
    objective = MockObjective(lambda x: x, points)

    result_ids = choose_ids(3, points, objective)

    assert set(points[result_ids]) == set([0, 1, 3])


def test_choose_ids_2():
    points = np.array([1, 5, 4, 3, 0])
    objective = MockObjective(lambda x: -x, points)

    result_ids = choose_ids(4, points, objective)

    assert set(points[result_ids]) == set([1, 3, 4, 5])


def test_chood_ids_all():
    points = np.array([1, 2, 3])
    objective = MockObjective(lambda _: None, points)

    assert set(choose_ids(3, points, objective)) == set([0, 1, 2])
    assert set(choose_ids(121, points, objective)) == set([0, 1, 2])


def test_quadratic_angles():
    a = np.array(
        [
            [-135.81935278, 634.8777047, 338.16458719],
            [-93.35100939, 634.8777047, 398.88488127],
            [-135.81935278, 637.80281964, 343.68587687],
        ]
    )

    b = np.array(
        [[-50.99345431, 643.70363649, 337.02070683], [-135.81935278, 634.8777047, 350.25115932]]
    )

    mat = quadratic_angles(a, b)

    expected_result = np.array(
        [[0.92704703, 0.99077017], [0.94045593, 0.9477817], [0.92760648, 0.9938237]]
    )

    assert np.all((-1.0 <= mat) & (mat <= 1.0))
    assert np.allclose(expected_result, mat)


def test_objective():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    center = np.zeros(3)

    fixed_points = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    obj1 = Objective(center, points, fixed_points=fixed_points)

    assert np.isclose(obj1(0, []), 0.3859375)
    assert np.isclose(obj1(0, [1, 2]), 0.31171875)

    obj2 = Objective(center, points, fixed_points=None)

    assert np.isclose(obj2(1, [1]), 0.95)


def test_orientations_from_domain():
    soma_center = np.array([-92.36826886, 645.07314376, 365.45216845])

    points = np.array(
        [
            [-135.81935278, 634.8777047, 338.16458719],
            [-93.35100939, 634.8777047, 398.88488127],
            [-135.81935278, 637.80281964, 343.68587687],
            [-50.99345431, 643.70363649, 337.02070683],
            [-135.81935278, 634.8777047, 350.25115932],
            [-83.80092951, 661.90131722, 378.25529382],
            [-95.24677112, 644.7095765, 397.33367781],
            [-115.0330315, 634.8777047, 385.67822932],
            [-88.08243399, 634.8777047, 328.14557669],
            [-42.21297597, 634.8777047, 342.18542843],
            [-41.09854123, 634.8777047, 338.42435016],
            [-41.48688437, 637.91957532, 338.60170204],
            [-45.34261913, 646.59918415, 341.30019129],
            [-42.21567796, 641.04743702, 339.89599549],
        ]
    )

    triangles = np.array(
        [
            [6, 7, 1],
            [10, 9, 1],
            [8, 10, 1],
            [0, 8, 1],
            [4, 0, 1],
            [7, 4, 1],
            [5, 6, 1],
            [12, 5, 1],
            [13, 12, 1],
            [9, 13, 1],
            [8, 0, 2],
            [3, 8, 2],
            [12, 3, 2],
            [5, 12, 2],
            [0, 4, 2],
            [6, 5, 2],
            [7, 6, 2],
            [4, 7, 2],
            [10, 8, 3],
            [11, 10, 3],
            [13, 11, 3],
            [12, 13, 3],
            [11, 13, 9],
            [10, 11, 9],
        ]
    )

    endfeet_targets = np.array([[-112.95478877, 635.31983868, 328.86005447]])

    orientations, lengths = orientations_from_domain(
        soma_center, points, triangles, 5, fixed_targets=endfeet_targets
    )

    expected_orientations = np.array(
        [
            [0.42830737, 0.15533069, 0.89018266],
            [0.87122951, -0.1732517, -0.45928531],
            [-0.89080984, -0.23020064, 0.39174672],
            [0.37551975, 0.73760491, 0.56118082],
            [-0.8306125, -0.19489638, -0.5216304],
        ]
    )

    expected_lengths = np.array([20.06971059, 58.84755654, 25.81708819, 22.81461691, 52.31209958])

    assert np.allclose(orientations, expected_orientations)
    assert np.allclose(expected_lengths, lengths)
