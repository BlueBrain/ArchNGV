from collections import deque

import morphio
import numpy as np
import pytest
from numpy import testing as npt

from archngv.app.utils import random_generator
from archngv.building.morphology_synthesis import perimeters as _p


@pytest.fixture
def parameters():
    return {
        "statistical_model": {"slope": 2.0, "intercept": 1.5, "standard_deviation": 0.01},
        "smoothing": {"window": [1.0, 1.0, 1.0, 1.0, 1.0]},
    }


@pytest.fixture
def statistical_model_parameters(parameters):
    return parameters["statistical_model"]


@pytest.fixture
def lrn_model(statistical_model_parameters):
    """linear regression with noise model"""
    rng = random_generator(0)
    return _p.LinearRegressionNoiseModel(statistical_model_parameters, rng=rng)


def test_linear_regression_noise_model(statistical_model_parameters, lrn_model):
    standard_deviation = statistical_model_parameters["standard_deviation"]

    xs = np.arange(100)

    ys = lrn_model.predict(xs[:, None])

    ys_without_noise = lrn_model._linear_function(xs[:, None])

    sdev = np.std(ys - ys_without_noise)
    npt.assert_allclose(sdev, standard_deviation, rtol=1e-1)


@pytest.fixture
def morphology():
    """6 [6, 8]
                         |
             1 [3, 4, 5, 6]
                |        |
                |     5 [6, 9, 10, 11, 12, 13, 14]]
                |
    0 [0, 1, 2, 3]
                |
                |     4 [5, 3, 2]
                |        |
             2 [3, 2, 1, 5]
                         |
                      3 [5, 6, 9, 8, 3, 2]
    """
    m = morphio.mut.Morphology()
    s_type = morphio.SectionType.basal_dendrite

    s0 = m.append_root_section(
        morphio.PointLevel(
            np.zeros((4, 3)).tolist(),  # points
            [1.0, 2.0, 3.0, 4.0],  # diameters
            [0.0, 1.0, 2.0, 3.0],
        ),  # perimeters
        s_type,
    )

    s1 = s0.append_section(
        morphio.PointLevel(
            np.zeros((4, 3)).tolist(),  # points
            [4.0, 5.0, 6.0, 7.0],  # diameters
            [3.0, 4.0, 5.0, 6.0],
        ),  # perimeters
        s_type,
    )

    s2 = s0.append_section(
        morphio.PointLevel(
            np.zeros((4, 3)).tolist(),  # points
            [4.0, 8.0, 9.0, 9.0],  # diameters
            [3.0, 2.0, 1.0, 5.0],
        ),  # perimeters
        s_type,
    )

    s3 = s2.append_section(
        morphio.PointLevel(
            np.zeros((6, 3)).tolist(),  # points
            [9.0, 5.0, 2.0, 3.0, 1.0, 4.0],  # diameters
            [5.0, 6.0, 9.0, 8.0, 3.0, 2.0],
        ),  # perimeters
        s_type,
    )

    s4 = s2.append_section(
        morphio.PointLevel(
            np.zeros((3, 3)).tolist(), [9.0, 5.0, 1.0], [5.0, 3.0, 2.0]  # points  # diameters
        ),  # perimeters
        s_type,
    )

    s5 = s1.append_section(
        morphio.PointLevel(
            np.zeros((7, 3)).tolist(),  # points
            [7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0],  # diameters
            [6.0, 9.0, 10, 11.0, 12, 13, 14],
        ),  # perimeters
        s_type,
    )

    s6 = s1.append_section(
        morphio.PointLevel(
            np.zeros((2, 3)).tolist(), [7.0, 1.0], [6.0, 8.0]  # points  # diameters
        ),  # perimeters
        s_type,
    )

    return m


def test_perimeters_upstream(morphology):
    """6 [6, 8]
                         |
             1 [3, 4, 5, 6]
                |        |
                |     5 [6, 9, 10, 11, 12, 13, 14]]
                |
    0 [0, 1, 2, 3]
                |
                |     4 [5, 3, 2]
                |        |
             2 [3, 2, 1, 5]
                         |
                      3 [5, 6, 9, 8, 3, 2]
    """
    sections = morphology.sections

    perimeters_upstream_list = [
        [],
        [2, 1, 0],
        [2, 1, 0],
        [1, 2, 3, 2, 1, 0],
        [1, 2, 3, 2, 1, 0],
        [5, 4, 3, 2, 1, 0],
        [5, 4, 3, 2, 1, 0],
    ]

    for section_id in range(7):
        expected_perimeters = perimeters_upstream_list[section_id]
        perimeters = np.fromiter(_p._perimeters_upstream(sections[section_id]), dtype=np.float32)
        npt.assert_array_equal(perimeters, expected_perimeters, err_msg=f"Section id: {section_id}")


def test_longest_downstream_leaf(morphology):
    sections = morphology.sections
    expected_leaves = [5, 5, 3, 3, 4, 5, 6]
    for section_id, expected_leaf in enumerate(expected_leaves):
        leaf = _p._longest_downstream_leaf(sections[section_id])
        npt.assert_equal(leaf.id, expected_leaf)


def test_longest_downstream_path(morphology):
    sections = morphology.sections

    expected_downstreams = [[1, 5], [5], [3], [], [], [], []]

    for section_id, expected_downstream in enumerate(expected_downstreams):
        path = _p._longest_downstream_path(sections[section_id])
        npt.assert_array_equal([s.id for s in path], expected_downstream)


def test_perimeters_downstream(morphology):
    sections = morphology.sections

    expected_perimeters_list = [
        [4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [6.0, 9.0, 8.0, 3.0, 2.0],
        [],
        [],
        [],
        [],
    ]

    for section_id in range(7):
        expected_perimeters = expected_perimeters_list[section_id]
        perimeters = np.fromiter(_p._perimeters_downstream(sections[section_id]), dtype=np.float32)
        npt.assert_allclose(perimeters, expected_perimeters, err_msg=f"Section id: {section_id}")


def test_array_from_generator():
    value_generator = iter(range(100))
    array_size = 7

    values = _p._array_from_generator(array_size, value_generator)
    npt.assert_allclose(values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6])

    value_generator = iter(range(5))
    array_size = 8

    values = _p._array_from_generator(array_size, value_generator)
    npt.assert_allclose(values, [0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0])

    value_generator = iter(range(0))
    array_size = 3

    with pytest.raises(StopIteration):
        values = _p._array_from_generator(array_size, value_generator)


def test_expand_start(morphology):
    """6 [6, 8]
                         |
             1 [3, 4, 5, 6]
                |        |
                |     5 [6, 9, 10, 11, 12, 13, 14]]
                |
    0 [0, 1, 2, 3]
                |
                |     4 [5, 3, 2]
                |        |
             2 [3, 2, 1, 5]
                         |
                      3 [5, 6, 9, 8, 3, 2]
    """
    sections = morphology.sections

    perimeters = _p._expand_start(sections[0], expansion_length=3)
    npt.assert_allclose(perimeters, [3, 2, 1])

    perimeters = _p._expand_start(sections[0], expansion_length=5)
    npt.assert_allclose(perimeters, [3, 3, 3, 2, 1])

    perimeters = _p._expand_start(sections[1], expansion_length=3)
    npt.assert_allclose(perimeters, [0.0, 1.0, 2.0])

    perimeters = _p._expand_start(sections[1], expansion_length=5)
    npt.assert_allclose(perimeters, [0.0, 0.0, 0.0, 1.0, 2.0])

    perimeters = _p._expand_start(sections[2], expansion_length=5)
    npt.assert_allclose(perimeters, [0.0, 0.0, 0.0, 1.0, 2.0])

    perimeters = _p._expand_start(sections[3], expansion_length=1)
    npt.assert_allclose(perimeters, [1.0])

    perimeters = _p._expand_start(sections[3], expansion_length=5)
    npt.assert_allclose(perimeters, [1.0, 2.0, 3.0, 2.0, 1.0])

    perimeters = _p._expand_start(sections[4], expansion_length=7)
    npt.assert_allclose(perimeters, [0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])


def test_expand_end(morphology):
    """6 [6, 8]
                         |
             1 [3, 4, 5, 6]
                |        |
                |     5 [6, 9, 10, 11, 12, 13, 14]]
                |
    0 [0, 1, 2, 3]
                |
                |     4 [5, 3, 2]
                |        |
             2 [3, 2, 1, 5]
                         |
                      3 [5, 6, 9, 8, 3, 2]
    """
    sections = morphology.sections

    perimeters = _p._expand_end(sections[0], expansion_length=1)
    npt.assert_allclose(perimeters, [4])

    perimeters = _p._expand_end(sections[0], expansion_length=3)
    npt.assert_allclose(perimeters, [4, 5, 6])

    perimeters = _p._expand_end(sections[0], expansion_length=5)
    npt.assert_allclose(perimeters, [4, 5, 6, 9, 10])

    perimeters = _p._expand_end(sections[2], expansion_length=1)
    npt.assert_allclose(perimeters, [6])

    perimeters = _p._expand_end(sections[2], expansion_length=3)
    npt.assert_allclose(perimeters, [6.0, 9.0, 8.0])

    perimeters = _p._expand_end(sections[2], expansion_length=5)
    npt.assert_allclose(perimeters, [6.0, 9.0, 8.0, 3.0, 2.0])

    perimeters = _p._expand_end(sections[2], expansion_length=7)
    npt.assert_allclose(perimeters, [6.0, 9.0, 8.0, 3.0, 2.0, 2.0, 2.0])

    perimeters = _p._expand_end(sections[4], expansion_length=1)
    npt.assert_allclose(perimeters, [2.0])

    perimeters = _p._expand_end(sections[6], expansion_length=2)
    npt.assert_allclose(perimeters, [8.0, 8.0])


def _predict_perimeters(morphology, lrn_model):
    sections = morphology.sections

    # set noise to zero
    lrn_model._norm = lambda value: 0.0

    for i in range(7):
        section = sections[i]
        perimeters = _p._predict_perimeters(section, lrn_model)
        expected_perimeters = section.diameters * lrn_model.slope + lrn_model.intercept
        npt.assert_allclose(perimeters, expected_perimeters)


def _smooth_perimeters(morphology, lrn_model):
    sections = morphology.sections

    # this window should not change the values
    smoothing_window = np.array([0.0, 1.0, 0.0])

    for i in range(7):
        section = sections[i]
        perimeters = _p._smooth_perimeters(section, smoothing_window)
        npt.assert_allclose(perimeters, section.perimeters)

    smoothing_window = np.array([1.0, 1.0, 1.0]) / 3.0

    perimeters_list = [
        [0.66666667, 1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0, 6.66666667],
        [2.33333333, 2.0, 2.66666667, 4.0],
        [4.0, 6.66666667, 7.66666667, 6.66666667, 4.33333333, 2.66666667],
        [3.0, 3.33333333, 2.66666667],
        [6.66666667, 8.33333333, 10.0, 11.0, 12.0, 13.0, 13.33333333],
        [6.33333333, 6.66666667],
    ]

    for i in range(7):
        section = sections[i]
        perimeters = _p._smooth_perimeters(section, smoothing_window)
        npt.assert_allclose(perimeters, perimeters_list[i])

    smoothing_window = np.ones(5) / 5.0

    section = sections[6]
    perimeters = _p._smooth_perimeters(section, smoothing_window)

    # expanded section 6: [4., 5., 6., 8., 6., 6.]
    expected_perimeters = [5.8, 6.2]
    npt.assert_allclose(perimeters, expected_perimeters)


def test_smooth_morphology_perimeters(morphology):
    smoothing_window = np.array([0.0, 2.0, 0.0])

    perimeters_list = [s.perimeters for s in morphology.iter()]

    _p._smooth_morphology_perimeters(morphology, smoothing_window)

    # all perimeters are multiplied by 2 due to the window, but divided
    # in the end by two because of the total scale
    for section, old_perimeters in zip(morphology.iter(), perimeters_list):
        npt.assert_allclose(section.perimeters, old_perimeters)


def test_add_perimeters_to_morphology(morphology, parameters):
    parameters["statistical_model"]["standard_deviation"] = 0.0
    parameters["smoothing"]["window"] = [0.0, 1.0, 0.0]

    slope = parameters["statistical_model"]["slope"]
    intercept = parameters["statistical_model"]["intercept"]

    _p.add_perimeters_to_morphology(morphology, parameters, rng=random_generator(seed=0))

    for section in morphology.iter():
        npt.assert_allclose(section.perimeters, section.diameters * slope + intercept)
        npt.assert_equal(len(section.points), len(section.perimeters))
        npt.assert_equal(len(section.diameters), len(section.perimeters))

        if not section.is_root:
            npt.assert_allclose(section.perimeters[0], section.parent.perimeters[-1])
