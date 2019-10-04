import morphio
import numpy as np
import pytest
from unittest.mock import Mock

from archngv.building.morphology_synthesis import endfoot_compartment as ec


def points():
    return np.array([[0.39837784, 0.26220273, 0.063706  ],
                     [0.02642341, 0.83682305, 0.1745578 ],
                     [0.60209137, 0.4267205 , 0.99843931],
                     [0.93499858, 0.89701073, 0.08564404],
                     [0.2880215 , 0.91225645, 0.78189027],
                     [0.68814504, 0.46642128, 0.49170066],
                     [0.14743094, 0.29872451, 0.23291819],
                     [0.06752298, 0.80174692, 0.55350953],
                     [0.71539294, 0.23468221, 0.41481889],
                     [0.69081039, 0.85747751, 0.51872049]])


class MockSection:

    def __init__(self):
        self.type = 2
        self.children = []
        self.id = 0

    @property
    def points(self):
        return points()

    def append_section(self, point_level, section_type):
        self.children.append((point_level, section_type))


class MockMorphology:
    def __init__(self):
        self.sections = [MockSection()]

    def iter(self):
        return iter(self.sections)

    def section(self, section_id):
        return self.sections[section_id]


@pytest.fixture
def endfeet_data():

    mock_mesh = Mock()
    mock_mesh.area = 2.
    mock_mesh.points = points()
    mock_mesh.triangles = [[0, 1, 2]]
    mock_mesh.thickness = 0.3

    mock_endfeet_data = Mock()
    mock_endfeet_data.area_meshes = [mock_mesh]
    mock_endfeet_data.targets = [np.random.random(3)]

    return mock_endfeet_data


def test_principal_direction_and_extents():

    principal_direction, centroid, left_extent, right_extent = \
        ec._principal_direction(points())

    expected_direction = (-0.94319842,  0.32081538, -0.08633785)
    expected_left_extent = 0.3602942167490586
    expected_right_extent = 0.5034604220178519
    expected_centroid = (0.4559215, 0.59940659, 0.43159052)

    assert np.allclose(expected_centroid, centroid)
    assert np.isclose(left_extent, expected_left_extent)
    assert np.isclose(right_extent, expected_right_extent)
    assert np.allclose(principal_direction, expected_direction)


def test_target_to_maximal_extent():

    ps = points()

    target = np.array([0.1, 0.1, 0.1], dtype=np.float)

    direction, extent = ec._target_to_maximal_extent(ps, target)

    expected_direction = (0.79654393, 0.43942254, 0.41524161)
    expected_length = 0.873461474423381

    assert np.allclose(direction, expected_direction)
    assert np.isclose(expected_length, extent)

    # These are the left and right extent points from the principal
    # direction
    target1 = np.array([0.79575044, 0.48381866, 0.46269755])
    target2 = np.array([-0.01894157,  0.76092444,  0.38812283])

    direction, extent = ec._target_to_maximal_extent(ps, target1)

    # thus we should get the opposite point as a result
    expected_direction = target2 - target1
    expected_length = np.linalg.norm(expected_direction)
    expected_direction /= expected_length

    assert np.allclose(expected_direction, direction)
    assert np.allclose(expected_length, extent)

    direction, extent = ec._target_to_maximal_extent(ps, target2)

    # thus we should get the opposite point as a result
    expected_direction = target1 - target2
    expected_length = np.linalg.norm(expected_direction)
    expected_direction /= expected_length

    assert np.allclose(expected_direction, direction)
    assert np.allclose(expected_length, extent)


def test_stump_section():

    # the actual data is not important
    start_point = section_direction = np.array([0.1, -0.5, 11.2])
    section_direction /= np.linalg.norm(section_direction)

    points, diameters, perimeters = ec._stump_section(start_point, section_direction)

    expected_points = [start_point,
                       start_point + 0.01 * section_direction,
                       start_point + 0.02 * section_direction,
                       start_point + 0.03 * section_direction]

    expected_diameters = expected_perimeters = [0.01, 0.01, 0.01, 0.01]

    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_allclose(diameters, expected_diameters)
    np.testing.assert_allclose(perimeters, expected_perimeters)


def test_endfoot_section():

    start_point = section_direction = np.array([0.1212, -1.1435, 10.5])
    section_direction /= np.linalg.norm(section_direction)

    total_area = 10.32
    total_volume = 31.5
    total_length = 2.4

    points, diameters, perimeters = ec._endfoot_section(start_point, total_area, total_volume, total_length, section_direction)

    expected_points = [
              start_point,
              start_point + total_length * 0.33333333 * section_direction,
              start_point + total_length * 0.66666667 * section_direction,
              start_point + total_length * section_direction]

    expected_diameters = [2.0 * np.sqrt(total_volume / (np.pi * total_length))] * 4
    expected_perimeters = [total_area / (np.pi * total_length)] * 4

    np.testing.assert_allclose(expected_points, points)
    np.testing.assert_allclose(expected_diameters, diameters)
    np.testing.assert_allclose(expected_perimeters, perimeters)

    # we need to reconstruct the initial area and volume

    points = np.asarray(points)
    diameters = np.asarray(diameters)
    perimeters = np.asarray(perimeters)

    lengths = np.linalg.norm(points[:1] - points[:-1], axis=1)
    reconstructed_volume = np.sum(lengths * (diameters[1:] * 0.5) ** 2 * np.pi)
    reconstructed_area = np.sum(lengths * perimeters[1:] * np.pi)

    assert np.isclose(total_volume, reconstructed_volume)
    assert np.isclose(total_area, reconstructed_area)


def test_add_compartments():

    section = MockSection()

    target = np.random.random(3)

    volume, area = 2., 5.

    ec._add_compartments(section, points(), target, volume, area)
    assert len(section.children) == 2


def test_add_endfeet_comparments(endfeet_data):

    mutable_morphology = MockMorphology()
    ec.add_endfeet_compartments(mutable_morphology, endfeet_data)
