from numpy.testing import assert_allclose
import utils
import numpy as np
from archngv.core.endfeet_area_reconstruction.detail import area_fitting, endfoot


def test_area_fit():
    world_offset = 111
    height = width = 10
    ef = endfoot.create_endfoot_from_global_data(
        0, *utils.create_mesh_data(height, width, world_offset))
    ef.extra = {'vertex':
                {'travel_times': np.arange(world_offset + ef.number_of_triangles,
                                           dtype=np.float32),
                 },
                }
    area_fitting.fit_area(ef, target_area=81 / 2.)

    assert ef.number_of_vertices == 49
    assert ef.number_of_triangles == 70
    assert ef.area == 35.0
    assert ef.local_to_global_map == {i: world_offset + i
                                      for i in range(ef.number_of_vertices)}
    assert_allclose(ef.vasculature_vertices,
                    [world_offset + i for i in range(ef.number_of_vertices)])

