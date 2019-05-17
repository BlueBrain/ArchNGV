import logging

import numpy as np
import stl.mesh


L = logging.getLogger(__name__)


def export_endfoot_mesh(endfoot, filepath):
    """ Exports either all the faces of the laguerre cells separately or as one object in stl format
    """
    triangles = endfoot.triangle_array

    try:
        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))

        cell_mesh.vectors = endfoot.coordinates_array[triangles]

        cell_mesh.save(filepath)

        L.info('Endfoot saved at: %s', filepath)

    except IndexError:
        msg = 'No triangles fro endfoot {}'.format(endfoot.filepath)
        L.error(msg)
        raise IndexError(msg)


def export_joined_endfeet_meshes(endfoot_iterator, filepath):

    vectors = np.array([
        triangle.tolist()
        for endfoot in endfoot_iterator
        for triangle in endfoot.coordinates[endfoot.triangles]
    ])

    cell_mesh = stl.mesh.Mesh(np.zeros(len(vectors), dtype=stl.mesh.Mesh.dtype))

    cell_mesh.vectors = vectors

    cell_mesh.save(filepath)
