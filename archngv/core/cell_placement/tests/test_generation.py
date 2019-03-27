from .. import generation

class MockVoxelData:

    def in_geometry(self, position):
        return False

@pytest.fixture
def placement_parameters():

    beta = 2.0
    number_of_trials = 120
    cutoff_radius = 2.0
    initial_sample_size = 1000

    return generation.PlacementParameters(beta,
                                          number_of_trials,
                                          cutoff_radius,
                                          initial_sample_size)

@pytest.fixture
def placement_generator():

    params = placement_parameters()
    return generation.PlacementGenerator()


def test_placement_parameters():

    params = placement_parameters()

    assert params.beta == beta
    assert params.number_of_trials == number_of_trials
    assert params.cutoff_radius == cutoff_radius
    assert params.initial_sample_size == initial_sample_size


def test_placement_generator_constructor():

    generation.PlacementGenerator(None, None, None, None, None, None)


def test_placement_generator_is_colliding__voxel_data(placement_generator):

    # patch bla bla

    test_point = np.array([1, 2, 3])

    assert placement_generator.is_colliding(test_point)


def test_placement_generator_is_colliding__empty_index_list(placement_generator):

    test_point = np.array([1, 2, 3])

    assert not placement_generator.is_colliding(test_point)


def test_placement_generator_is_colliding__full_index_list(placement_generator):

    test_point = np.array([1, 2, 3])

    assert placement_generator.is_colliding(test_point)


def test_placement_generator_first_order():
    pass


def test_placement_generator_second_order():
    pass


def test_generator_run():
    pass
