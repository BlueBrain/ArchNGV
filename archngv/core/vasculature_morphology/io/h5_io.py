import h5py
import numpy as np


def h5_loader(filename):
    """ Load an hdf5 vasculature file. The datasets points, edges, radii and types are expected
    """

    if not filename.endswith('.h5'):
        filename += '.h5'

    with h5py.File(filename, 'r') as F:
        points = F['points'][:]
        edges = F['edges'][:]
        radii = F['radii'][:]
        types = F['types'][:]

    return points, edges, radii, types 


def h5_writer(filename, points, edges, radii, types):
    """ Writes the given points, edges, radii and types to an hdf5 file by creating
    the respective datasets
    """

    if not filename.endswith('.h5'):
        filename += '.h5'

    with h5py.File(filename, 'w') as F:

        F.create_dataset(name='points', data=points)
        F.create_dataset(name='edges', data=edges, dtype=np.intp)
        F.create_dataset(name='radii', data=radii)
        F.create_dataset(name='types', data=types)
