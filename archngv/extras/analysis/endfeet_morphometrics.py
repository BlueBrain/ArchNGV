'''analysis of endfeet'''
import logging
import os

import numpy as np
from scipy.spatial import cKDTree
import morphio

from archngv.utils.ngons import vectorized_triangle_area

from archngv.extras.analysis.common import find_layer

L = logging.getLogger(__name__)


def endfeet_morphometrics(ngv_circuit, astrocyte_ids):
    '''endfeet_morphometrics'''
    data = {'path_lengths': [],
            'areas': [],
            'layers': [],
            'distances': []}

    endfeetome = ngv_circuit.data.endfeetome

    for astrocyte_id in astrocyte_ids:
        astrocyte_path = os.path.join(
            ngv_circuit.config.morphology_directory,
            ngv_circuit.data.astrocytes.astrocyte_names[astrocyte_id].decode('utf-8') + '.h5'
        )

        endfeet_ids = ngv_circuit.connectome.gliovascular.astrocyte.to_endfoot(astrocyte_id)
        if len(endfeet_ids) > 0:
            current_endfeet_targets = endfeetome.targets.endfoot_surface_coordinates[endfeet_ids]

            for endfoot_section in find_endfoot_section(morphio.Morphology(astrocyte_path),
                                                        current_endfeet_targets):
                path_length, distance = path_length_distance_to_root(endfoot_section)
                data['path_lengths'].append(path_length)
                data['distances'].append(distance)

            for endfoot_id in endfeet_ids:
                endfoot = endfeetome.surface_meshes[endfoot_id]
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
