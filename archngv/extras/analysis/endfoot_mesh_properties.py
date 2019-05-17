import sys
import numpy as np
from morphmath import vectorized_triangle_area

from archngv.core.data_structures import NGVConfig
from archngv.core.data_structures import NGVCircuit


def calculate_mesh_area(endfoot_mesh):
    """ Given an endfoot mesh, extract its total area
    """
    points = endfoot_mesh.points
    triangles = endfoot_mesh.triangles

    vectors1 = points[triangles[:, 1]] - points[triangles[:, 0]]
    vectors2 = points[triangles[:, 2]] - points[triangles[:, 0]]

    return np.sum(vectorized_triangle_area(vectors1, vectors2))


def extract_endfeet_mesh_properties(ngv_config_path):
    """ Extract endfeet mesh properties.
    """
    config = NGVConfig.from_file(ngv_config_path)
    endfeetome = NGVCircuit(config).data.endfeetome

    dict_data = {}

    mesh_areas = np.zeros(len(endfeetome), dtype=np.float)

    for i in range(len(endfeetome)):

        try:

            mesh = endfeetome[i]

            mesh_areas[i] = calculate_mesh_area(mesh)

        except KeyError:
            L.warning('Endfoot %d has no mesh.', i)

    dict_data['areas'] = mesh_areas

    return dict_data


if __name__ == '__main__':

    import json

    circuit_config_path = sys.argv[1]
    output_path = sys.argv[2]

    dict_data = \
        extract_endfeet_mesh_properties(circuit_config_path)

    with open(output_path, 'w') as fd:
        json.dump(fd, output_path, indent=4)
