from random import shuffle

import numpy as np
import pandas as pd

from archngv.building.connectivity.detail.gliovascular_generation.graph_reachout import (
    _argsort_components,
    _distribute_elements_in_buckets,
    _maximum_reachout,
    _select_component_targets,
)


def test__distribute_elements_in_buckets():
    res = _distribute_elements_in_buckets(11, np.array([2, 5, 9]))
    np.testing.assert_allclose(res, [2, 5, 4])
    np.testing.assert_allclose(res.sum(), 11)

    res = _distribute_elements_in_buckets(11, np.array([10, 3, 3]))
    np.testing.assert_allclose(res, [5, 3, 3])
    np.testing.assert_allclose(res.sum(), 11)

    res = _distribute_elements_in_buckets(2, np.array([1, 1, 1]))
    np.testing.assert_allclose(res, [1, 1, 0])
    np.testing.assert_allclose(res.sum(), 2)

    res = _distribute_elements_in_buckets(100, np.array([50, 50]))
    np.testing.assert_allclose(res, [50, 50])
    np.testing.assert_allclose(res.sum(), 100)

    res = _distribute_elements_in_buckets(100, np.array([50, 50]))
    np.testing.assert_allclose(res, [50, 50])
    np.testing.assert_allclose(res.sum(), 100)

    res = _distribute_elements_in_buckets(100, np.array([50, 50, 1]))
    np.testing.assert_allclose(res, [50, 49, 1])
    np.testing.assert_allclose(res.sum(), 100)

    res = _distribute_elements_in_buckets(100, np.array([50, 1, 50, 20]))
    np.testing.assert_allclose(res, [50, 1, 29, 20])
    np.testing.assert_allclose(res.sum(), 100)

    res = _distribute_elements_in_buckets(1, np.array([1, 0, 0]))
    np.testing.assert_allclose(res, [1, 0, 0])
    np.testing.assert_allclose(res.sum(), 1)

    res = _distribute_elements_in_buckets(1, np.array([0, 1, 0]))
    np.testing.assert_allclose(res, [0, 1, 0])
    np.testing.assert_allclose(res.sum(), 1)

    res = _distribute_elements_in_buckets(1, np.array([0, 0, 1]))
    np.testing.assert_allclose(res, [0, 0, 1])
    np.testing.assert_allclose(res.sum(), 1)

    res = _distribute_elements_in_buckets(2, np.array([1, 0, 1]))
    np.testing.assert_allclose(res, [1, 0, 1])
    np.testing.assert_allclose(res.sum(), 2)

    res = _distribute_elements_in_buckets(2, np.array([0, 2, 0]))
    np.testing.assert_allclose(res, [0, 2, 0])
    np.testing.assert_allclose(res.sum(), 2)

    res = _distribute_elements_in_buckets(13, np.array([10, 0, 5]))
    np.testing.assert_allclose(res, [9, 0, 4])
    np.testing.assert_allclose(res.sum(), 13)


def test_select_component_targets__point_line():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [2.01, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [6.5, 0.0, 0.0],
        ]
    )

    source = np.array([3.0, -1.0, 0.0])

    component = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})

    ids = _select_component_targets(source, component, 1)

    np.testing.assert_allclose(ids, [3])

    ids = _select_component_targets(source, component, 2)

    assert set(ids) == set([3, 6])

    ids = _select_component_targets(source, component, 3)

    assert set(ids) == set([0, 3, 6])

    ids = _select_component_targets(source, component, 4)

    assert set(ids) == set([0, 3, 4, 6])

    ids = _select_component_targets(source, component, 5)

    assert set(ids) == set([0, 1, 3, 4, 6])

    ids = _select_component_targets(source, component, 7)

    assert set(ids) == set(range(7))


def test_argsort_components():
    source = np.array([0.0, -1.0, 2.0])

    comps = []

    for i in range(5):
        points = np.column_stack((np.ones(10) * 2.0 + i, np.arange(-5.0, 5.0), np.zeros(10)))
        component = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})

        comps.append(component)

    shuffle_ids = [2, 0, 4, 1, 3]
    comps = [comps[i] for i in shuffle_ids]

    sorted_ids, closest_vertices = _argsort_components(source, comps)
    expected_ids = [1, 3, 0, 4, 2]

    assert np.allclose(closest_vertices, 4)
    assert np.allclose(sorted_ids, expected_ids)
