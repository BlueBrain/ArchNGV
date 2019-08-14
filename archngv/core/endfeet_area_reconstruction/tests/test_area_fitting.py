import utils
import numpy as np
from archngv.core.endfeet_area_reconstruction.detail import area_fitting, endfoot


def test_area_fit():
    ef = endfoot.create_endfoot_from_global_data(0,
                                                 *utils.create_mesh_data(height=10,
                                                                         width=10))
    ef.extra = {'vertex':
                {'travel_times': np.arange(ef.number_of_triangles,
                                           dtype=np.float32),
                 },
                }
    area_fitting.fit_area(ef, target_area=81 / 2.)

    assert ef.number_of_vertices == 49
    assert ef.number_of_triangles == 70
    assert ef.area == 35.0
    assert ef.local_to_global_map == {i: i for i in range(ef.number_of_vertices)}
