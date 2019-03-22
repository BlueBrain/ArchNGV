import os
import h5py
import itertools

import numpy


def export_structure(filename, tesselation, global_coordinate_system=False):

    n_cells = len(tesselation)

    neighbors = tesselation.connectivity
    flat_connectivity = [(index, n) for index, ns in enumerate(neighbors) for n in ns]

    with h5py.File(filename, 'w') as fd:

        data_group = fd.create_group('Data')

        data_points = []
        data_triangles = []
        data_neighbors = []

        offsets = numpy.zeros((n_cells + 1, 3), dtype=numpy.uintp)
        current_offsets = numpy.zeros(3, dtype=numpy.uintp)

        for index, cell in enumerate(tesselation):

            cell_points = cell.points
            cell_triangles = cell.triangles
            cell_neighbors = neighbors[index]

            # before the new offset
            if global_coordinate_system:
                data_triangles.extend(cell_triangles + current_offsets[0])
            else:
                data_triangles.extend(cell_triangles)

            current_offsets[0] += len(cell_points)
            current_offsets[1] += len(cell_triangles)
            current_offsets[2] += len(cell_neighbors)

            offsets[index + 1] = current_offsets

            data_points.extend(cell_points)
            data_neighbors.extend(cell_neighbors)

        data_group.create_dataset('points', data=data_points, dtype=numpy.float)
        data_group.create_dataset('triangles', data=data_triangles, dtype=numpy.uintp)
        data_group.create_dataset('neighbors', data=data_neighbors, dtype=numpy.uintp)

        offsets_dset = fd.create_dataset('offsets', data=offsets, dtype=numpy.uintp)
        offsets_dset.attrs["column_names"] = \
        numpy.array(['points', 'triangles', 'neighbors'], dtype=h5py.special_dtype(vlen=str))

        conn_dset = fd.create_dataset('connectivity', data=flat_connectivity, dtype=numpy.uintp)
        conn_dset.attrs["column_names"] = \
        numpy.array(['i_domain_index', 'j_domain_index'], dtype=h5py.special_dtype(vlen=str))

def export_mesh(tesselation, filepath):
    """ Exports either all the faces of the laguerre cells separately or as one object in stl format
    """
    import stl.mesh
    """
    def individual_mesh(cell_id, cell):

        coo_triangles = cell.points[cell.triangles]

        cell_mesh = stl.mesh.Mesh(numpy.zeros(len(coo_triangles), dtype=stl.mesh.Mesh.dtype))

        cell_mesh.vectors = coo_triangles

        cell_mesh.save(path)

        return coo_triangles
    """
    #if joined_mesh_path is None:
    #    map(lambda tup: individual_mesh(*tup), enumerate(tesselation))
    #else:
    triangles = [triangle for _, cell in enumerate(tesselation) for triangle in cell.points[cell.triangles]]

    cell_mesh = stl.mesh.Mesh(numpy.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))

    cell_mesh.vectors = numpy.asarray(triangles, dtype=numpy.float)

    cell_mesh.save(filepath)
