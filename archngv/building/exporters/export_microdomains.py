""" Microdomain expoerters functions """

import h5py
import numpy


def export_structure(filename, domains):
    """ Export microdomain tesselation structure

    Args:
        domains: list[Microdomain]

    Notes:
        HDF5 Layout Hierarchy:
            data:
                points: array[float32, (N, 3)]
                    xyz coordinates of microdomain points
                triangle_data   array[uint64, (M, 4)]
                    [polygon_id, v0, v1, v2]
                    The polygon the triangle belongs to and its vertices
                neighbors
                    The neighbors to each triangle. Negative numbers signify a
                    bounding box wall.

            offsets: array[uint64, (N + 1, 3)]
                [points, triangle_data, neighbors]
                The data for the i-th domain:
                    points[offsets[i, 0]: offsets[i + 1, 0]]
                    triangle_data[offsets[i, 1]: offsets[i + 1, 1]]
                    neighbors[offsets[i, 2]: offsets[i + 1, 2]]
    """
    n_domains = len(domains)
    with h5py.File(filename, 'w') as fd:

        data_group = fd.create_group('data')
        points, triangle_data, neighbors = [], [], []

        # offsets are of size n_domains + 1 because it is convenient
        # to query the offset of the i-th domain as (offsets[i], offsets[i + 1])
        offsets = numpy.zeros((n_domains + 1, 3), dtype=numpy.uint64)

        for index, dom in enumerate(domains):

            ps, tri_data, neighs = dom.points, dom.triangle_data, dom.neighbor_ids

            offsets[index + 1] = offsets[index] + (len(ps), len(tri_data), len(neighs))

            points.extend(ps)
            triangle_data.extend(tri_data)
            neighbors.extend(neighs)

        data_group.create_dataset('points', data=points, dtype=numpy.float32)
        data_group.create_dataset('triangle_data', data=triangle_data, dtype=numpy.uint64)
        data_group.create_dataset('neighbors', data=neighbors, dtype=numpy.int64)

        offsets_dset = fd.create_dataset('offsets', data=offsets, dtype=numpy.uint64)
        offsets_dset.attrs["column_names"] = numpy.array(
            ['points', 'triangle_data', 'neighbors'],
            dtype=h5py.special_dtype(vlen=str)
        )
