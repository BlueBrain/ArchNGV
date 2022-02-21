""" Endfeetome exporters """
import logging

import h5py
import numpy as np

from archngv.exceptions import NGVError

L = logging.getLogger(__name__)


def _write_endfoot_layout(root, index, points, triangles):

    mesh_group = root.create_group("endfoot_{}".format(index))
    mesh_group.create_dataset("points", data=points, dtype=np.float32)
    mesh_group.create_dataset("triangles", data=triangles, dtype=np.uint64)


def export_endfeet_areas(filepath, data_generator, n_endfeet):
    """Endfeetome"""
    with h5py.File(filepath, "w") as fd:

        metadata = fd.create_group("metadata")
        metadata.attrs["object_type"] = "endfoot_mesh"

        meshes = fd.create_group("objects")
        attributes = fd.create_group("attributes")

        is_empty = np.ones(n_endfeet, dtype=bool)

        # datasets with 1D properties
        endfeet_areas = np.zeros(n_endfeet, dtype=np.float32)
        endfeet_areas_initial = np.zeros(n_endfeet, dtype=np.float32)
        endfeet_thicknesses = np.zeros(n_endfeet, dtype=np.float32)

        for (
            endfoot_index,
            points,
            triangles,
            initial_area,
            final_area,
            thickness,
        ) in data_generator:

            _write_endfoot_layout(
                root=meshes, index=endfoot_index, points=points, triangles=triangles
            )

            endfeet_areas[endfoot_index] = final_area
            endfeet_areas_initial[endfoot_index] = initial_area
            endfeet_thicknesses[endfoot_index] = thickness

            is_empty[endfoot_index] = False

        # empty placeholders
        empty_endfeet = np.where(is_empty)[0]
        for endfoot_index in empty_endfeet:

            _write_endfoot_layout(
                root=meshes,
                index=endfoot_index,
                points=np.empty([0, 3]),
                triangles=np.empty([0, 3]),
            )

            L.info("Endfoot %d is empty", endfoot_index)

        # write 1D datasets
        attributes.create_dataset("surface_area", data=endfeet_areas)
        attributes.create_dataset("unreduced_surface_area", data=endfeet_areas_initial)
        attributes.create_dataset("surface_thickness", data=endfeet_thicknesses)


def export_endfoot_mesh(endfoot_coordinates, endfoot_triangles, filepath):
    """Exports either all the faces of the laguerre cells separately or as one object
    in stl format"""
    import stl.mesh

    try:
        cell_mesh = stl.mesh.Mesh(np.zeros(len(endfoot_triangles), dtype=stl.mesh.Mesh.dtype))

        cell_mesh.vectors = endfoot_coordinates[endfoot_triangles]

        cell_mesh.save(filepath)

        L.info("Endfoot saved at: %s", filepath)

    except IndexError as e:
        msg = "No triangles found"
        L.error(msg)
        raise NGVError(msg) from e


def export_joined_endfeet_meshes(endfoot_iterator, filepath):
    """Exports the joined meshes TODO: fix this by replacing the endfoot iterato"""
    import stl.mesh

    vectors = np.array(
        [
            triangle.tolist()
            for endfoot in endfoot_iterator
            for triangle in endfoot.coordinates[endfoot.triangles]
        ]
    )

    cell_mesh = stl.mesh.Mesh(np.zeros(len(vectors), dtype=stl.mesh.Mesh.dtype))

    cell_mesh.vectors = vectors

    cell_mesh.save(filepath)
