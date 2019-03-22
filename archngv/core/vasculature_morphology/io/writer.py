import numpy as np
from time import ctime


def import_vtk_writer():
    """ Indirection for optional import of vtk
    """
    from .vtk_io import vtk_writer
    return vtk_writer


def import_h5_writer():
    """ Indirection for optional import of h5py
    """
    from .h5_io import h5_writer
    return h5_writer


_WRITERS = {'vtk': import_vtk_writer,
            'h5': import_h5_writer}

def save_vasculature(vasc, filename='vasculature', timestamp=False, mode='h5'):
    """ Saves the vasculature object with the specified file format
    """

    points = vasc.data.points
    edges = vasc.graph.edges
    radii = vasc.data.radii
    types = vasc.data.types

    # timestamp
    if timestamp:
        filename += '_' + ctime().replace(' ', '_')

    _WRITERS[mode]()(filename, points, edges, radii, types)
