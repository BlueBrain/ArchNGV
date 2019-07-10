import h5py
import logging
import numpy as np
from scipy import stats

L = logging.getLogger(__name__)


def create_endfeet_areas(ngv_config, map_func):

    from archngv.core.data_structures.data_gliovascular import GliovascularData
    from scipy.spatial import cKDTree
    import openmesh

    mesh_points = openmesh.read_trimesh(ngv_config.input_paths('vasculature_mesh')).points()

    with GliovascularData(ngv_config.output_paths('gliovascular_data')) as gdata:
        surface_endfeet_targets = gdata.endfoot_surface_coordinates[:]

    distances, indices = cKDTree(mesh_points).query(surface_endfeet_targets, k=1)

    unique_idx = np.unique(indices)

    reg = {index: [] for index  in unique_idx}


    for n, index in enumerate(indices):

        reg[index].append(n)

    problematic = [item[::-1] for item in reg.items() if len(item[1]) > 1]

    print(problematic)

if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config = NGVConfig.from_file(sys.argv[1])
    create_endfeet_areas(config, map)


