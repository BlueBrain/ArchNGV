from unittest.mock import Mock, PropertyMock, patch

import numpy as np
from voxcell import VoxelData

import archngv.building.cell_placement.generation as generation


class MockEnergy:
    def has_second_order_potentials(self):
        return False


class MockIntensity:
    voxel_dimensions = None
    raw = None
    shape = (10, 10, 10)
    voxel_volume = 1.0

    def indices_to_positions(self, val):
        return np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])


class MockVoxelData:
    def __init__(self, vox_int):
        self.voxelized_intensity = vox_int

    def in_geometry(self):
        pass


class MockSomaDistribution:
    def __call_(self):
        return None


def placement_parameters():
    beta = 2.0
    number_of_trials = 120
    cutoff_radius = 2.0
    initial_sample_size = 100

    return generation.PlacementParameters(
        beta, number_of_trials, cutoff_radius, initial_sample_size
    )


def placement_generator():
    parameters = placement_parameters()
    intensity = MockIntensity()
    voxel_data = MockVoxelData(intensity)
    energy_operator = MockEnergy()
    index_list = []
    total_spheres = 100
    soma_distribution = MockSomaDistribution()

    return generation.PlacementGenerator(
        parameters, total_spheres, voxel_data, energy_operator, index_list, soma_distribution
    )


def test_placement_parameters():
    params = placement_parameters()

    assert params.beta == 2.0
    assert params.number_of_trials == 120
    assert params.cutoff_radius == 2.0
    assert params.initial_sample_size == 100


def test_placement_generator_constructor():
    p_gen = placement_generator()

    assert isinstance(p_gen.vdata, MockVoxelData)
    assert isinstance(p_gen.energy_operator, MockEnergy)
    assert isinstance(p_gen.parameters, generation.PlacementParameters)
    assert p_gen._total_spheres == 100


def test_placement_generator__method_selection():
    with patch.object(MockEnergy, "has_second_order_potentials", return_value=True):
        p_gen = placement_generator()
        assert p_gen.method == p_gen.second_order

    with patch.object(MockEnergy, "has_second_order_potentials", return_value=False):
        p_gen = placement_generator()
        assert p_gen.method == p_gen.first_order


def test_placement_generator_is_colliding__voxel_data():
    test_point = np.array([1, 2, 3])
    test_radius = 2.0

    with patch.object(MockVoxelData, "in_geometry", return_value=False):
        p_gen = placement_generator()
        assert p_gen.is_colliding(test_point, test_radius)

    with patch.object(MockVoxelData, "in_geometry", return_value=True):
        p_gen = placement_generator()
        assert not p_gen.is_colliding(test_point, test_radius)


def test_placement_generator_is_colliding__empty_index_list():
    test_point = np.array([1, 2, 3])
    test_radius = 2.0

    with patch.object(MockVoxelData, "in_geometry", return_value=True):
        p_gen = placement_generator()

        mock_index = Mock(is_intersecting=lambda p, r: True)

        p_gen.index_list = [mock_index]

        assert p_gen.is_colliding(test_point, test_radius)


def test_placement_generator_is_colliding__pattern():
    test_point = np.array([1, 2, 3])
    test_radius = 2.0

    with patch.object(MockVoxelData, "in_geometry", return_value=True):
        p_gen = placement_generator()

        mock_index = Mock(is_intersecting=lambda p, r: False)
        p_gen.index_list = [mock_index]

        assert not p_gen.is_colliding(test_point, test_radius)

        # almost touching
        new_point = np.array([4.01, 2.0, 3.0])
        new_radius = 1.0
        p_gen.pattern.add(new_point, new_radius)

        assert not p_gen.is_colliding(test_point, test_radius)

        # touching
        new_point = np.array([-2.0, 2.0, 3.0])
        new_radius = 1.0
        p_gen.pattern.add(new_point, new_radius)

        assert p_gen.is_colliding(test_point, test_radius)


def test_placement_generator_first_order():
    voxel_centers = np.array([[0.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])

    with patch.object(
        MockIntensity, "voxel_dimensions", new_callable=PropertyMock, return_value=(1.0,)
    ), patch.object(MockSomaDistribution, "__call__", return_value=1.2), patch.object(
        MockVoxelData, "in_geometry", return_value=True
    ), patch.object(
        generation, "proposal", return_value=(1.0, 2.0, 3.0)
    ):
        p_gen = placement_generator()

        new_point, new_radius = p_gen.first_order(voxel_centers)

        assert np.allclose(new_point, (1.0, 2.0, 3.0))
        assert np.allclose(new_radius, 1.2)


def test_placement_generator_second_order():
    pass


def test_generator_run():
    mock_point = np.array([1.0, 2.0, 3.0])
    mock_radius = 1.4

    voxel_centers = np.array([[0.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])

    p_gen = placement_generator()

    with patch.object(p_gen, "method", return_value=(mock_point, mock_radius)), patch.object(
        generation, "nonzero_intensity_groups", return_value=((10, voxel_centers) for _ in range(2))
    ):
        p_gen.run()

        coordinates = p_gen.pattern.coordinates
        radii = p_gen.pattern.radii

        assert len(coordinates) == len(radii) == 20, (coordinates, radii)

        assert np.allclose(coordinates - mock_point, 0.0)
        assert np.allclose(radii, mock_radius)


def test_proposal():
    voxel_centers = np.array([[1.0, 1.0, 1.0]])
    voxel_edge_length = 0.0

    result_point = generation.proposal(voxel_centers, voxel_edge_length)

    assert np.allclose(result_point, voxel_centers[0])


def test_voxel_grid_centers():
    raw_array = np.zeros((2, 2, 2), dtype=np.float32)
    voxel_dimensions = (2, 2, 2)

    offset = (-1.0, 2.0, 3.0)

    voxel_data = VoxelData(raw_array, voxel_dimensions, offset)

    centers = generation.voxel_grid_centers(voxel_data)

    expected = np.array(
        [
            [0.0, 3.0, 4.0],
            [0.0, 3.0, 6.0],
            [0.0, 5.0, 4.0],
            [0.0, 5.0, 6.0],
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 6.0],
            [2.0, 5.0, 4.0],
            [2.0, 5.0, 6.0],
        ]
    )

    assert np.allclose(centers - expected, 0.0)


def test_voxel_group_centers():
    raw_array = np.zeros((2, 2, 2), dtype=np.float32)

    raw_array[..., 0] = 1e10
    raw_array[0, ...] = 2e10

    voxel_dimensions = (2, 2, 2)

    offset = (-1.0, 2.0, 3.0)

    voxel_data = VoxelData(raw_array, voxel_dimensions, offset)

    # group together voxels with identical values
    intensity_per_group, group_indices, voxels_per_group = np.unique(
        voxel_data.raw, return_inverse=True, return_counts=True
    )

    groups = generation.voxels_group_centers(group_indices, voxel_data)

    assert len(groups) == 3

    assert np.allclose(groups[0], np.array([[2.0, 3.0, 6.0], [2.0, 5.0, 6.0]]))
    assert np.allclose(groups[1], np.array([[2.0, 3.0, 4.0], [2.0, 5.0, 4.0]]))
    assert np.allclose(
        groups[2], np.array([[0.0, 3.0, 4.0], [0.0, 3.0, 6.0], [0.0, 5.0, 4.0], [0.0, 5.0, 6.0]])
    )


def test_counts_per_group():
    intensity_per_group = np.array([100000.0, 200000.0, 350000.0])
    voxels_per_group = np.array([100, 120, 210])
    voxel_volume = 1000.0

    results = generation.counts_per_group(intensity_per_group, voxels_per_group, voxel_volume)

    expected = np.array([10, 24, 73], dtype=np.int32)

    assert np.all(results == expected)


def test_nonzero_intensity_groups():
    raw_array = np.zeros((2, 2, 2), dtype=np.float32)

    raw_array[..., 0] = 1e10
    raw_array[0, ...] = 2e10

    voxel_dimensions = (2, 2, 2)

    offset = (-1.0, 2.0, 3.0)

    voxel_data = VoxelData(raw_array, voxel_dimensions, offset)

    results = list(generation.nonzero_intensity_groups(voxel_data))

    assert len(results) == 2

    res1, res2 = results

    assert res1[0] == 160
    assert np.allclose(res1[1], np.array([[2.0, 3.0, 4.0], [2.0, 5.0, 4.0]]))

    assert res2[0] == 640
    assert np.allclose(
        res2[1], np.array([[0.0, 3.0, 4.0], [0.0, 3.0, 6.0], [0.0, 5.0, 4.0], [0.0, 5.0, 6.0]])
    )
