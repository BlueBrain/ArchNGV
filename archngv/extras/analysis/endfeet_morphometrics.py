'''analysis of endfeet'''
import logging
import os

import numpy as np
from scipy.spatial import cKDTree
import morphio

from archngv.math_utils import vectorized_triangle_area


L = logging.getLogger(__name__)

# TODO: these need to be passed in as parameters
LAYERS = {'bins': np.array([0.0,
                            674.68206269999996,
                            1180.8844627000001,
                            1363.6375343,
                            1703.8656135000001,
                            1847.3347831999999,
                            2006.3482524000001]),
          'labels': ('VI', 'V', 'IV', 'III', 'II', 'I'),
          'centers': np.array([337.34103135,
                               927.7832627,
                               1272.2609985,
                               1533.7515739,
                               1775.60019835,
                               1926.8415178])
          }


def find_layer(y_coordinates):
    '''use `LAYERS` to find the layer for the `y_coordinates`'''
    layer = np.searchsorted(LAYERS['bins'], y_coordinates)

    if not np.all(1 <= layer <= 6):
        L.warning('Y coordinate %s exceeded layer max. It will be clipped', y_coordinates)

    return np.clip(layer, 1, 6)


def endfeet_morphometrics(ngv_circuit, astrocyte_ids):
    '''endfeet_morphometrics'''
    data = {'path_lengths': [],
            'areas': [],
            'layers': [],
            'distances': []}

    endfeet_areas = ngv_circuit.data.endfeetome.areas
    for astrocyte_id in astrocyte_ids:
        astrocyte_path = os.path.join(
            ngv_circuit.config.morphology_directory,
            ngv_circuit.data.astrocytes.astrocyte_names[astrocyte_id].decode('utf-8') + '.h5'
        )

        endfeet_ids = ngv_circuit.connectome.gliovascular.astrocyte.to_endfoot(astrocyte_id)
        if len(endfeet_ids) > 0:
            current_endfeet_targets = endfeet_areas.targets.endfoot_surface_coordinates[endfeet_ids]

            for endfoot_section in find_endfoot_section(morphio.Morphology(astrocyte_path),
                                                        current_endfeet_targets):
                path_length, distance = path_length_distance_to_root(endfoot_section)
                data['path_lengths'].append(path_length)
                data['distances'].append(distance)

            for endfoot_id in endfeet_ids:
                endfoot = endfeet_areas[endfoot_id]
                area = vectorized_triangle_area((endfoot.points[endfoot.triangles[:, 1]] -
                                                 endfoot.points[endfoot.triangles[:, 0]]),
                                                (endfoot.points[endfoot.triangles[:, 2]] -
                                                 endfoot.points[endfoot.triangles[:, 0]])
                                                )
                data['areas'].append(float(area.sum()))

            for endfoot_coo in current_endfeet_targets:
                data['layers'].append(int(find_layer(endfoot_coo[1])))

    return data


def find_endfoot_section(astrocyte, endfeet_targets):
    '''find endfoot section'''
    leaf_sections = [s for s in astrocyte.iter() if not s.children and s.type == 2]
    leaf_points = np.asarray([s.points[-1] for s in leaf_sections])
    _, indices = cKDTree(leaf_points).query(endfeet_targets)
    return [leaf_sections[i] for i in indices]


def path_length_distance_to_root(leaf_section):
    '''calculate path length distance to root'''
    current_section = leaf_section
    last_point = leaf_section.points[-1]
    path_length = 0.0

    while True:
        points = current_section.points
        path_length += np.linalg.norm(points[1:] - points[0:-1], axis=1).sum()

        try:
            current_section = current_section.parent
        except morphio.MissingParentError:
            first_point = current_section.points[0]
            distance = float(np.linalg.norm(last_point - first_point))
            break

    return path_length, distance
