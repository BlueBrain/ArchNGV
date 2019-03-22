import os
import logging
from .curation import curate
from ..vasculature import Vasculature
from .convert_to_spec import convert_to_spec


log = logging.getLogger(__name__)


def import_vtk_loader():
    """ Indirection for optional import of vtk
    """
    from .vtk_io import vtk_loader
    return vtk_loader


def import_h5_loader():
    """ Indirection for optional import of h5py
    """
    from .h5_io import h5_loader
    return h5_loader


_LOADERS = {'vtk': import_vtk_loader,
            'h5':  import_h5_loader}


def _deduce_mode(filename):

    if filename.endswith('.vtk'):

        mode = 'vtk'

    elif filename.endswith('.h5'):

        mode = 'h5'

    else:

        log.exception('File extension not recognized or not existent for file {}'.format(filename))
        raise TypeError 

    log.debug('Mode {} was deduced for file {}'.format(mode, filename))
    return mode


def load_vasculature(filename, mode=None):
    """ Loads the vasculature object from file with specified file format

    Warning: If the dataset has the radii on the segments and not on the nodes
    they are automatically converted into radii on the nodes.
    """
    # check if file exists and return meaningful message
    if not os.path.exists(filename):
        raise OSError("File {0} does not exist...".format(filename))

    mode = _deduce_mode(filename) if mode is None else mode

    points, edges, radii, types = _LOADERS[mode]()(filename)

    points, edges, radii, types = curate(points, edges, radii, types)

    point_data, segment_structure, chain_structure, chain_connectivity = \
    convert_to_spec(points, edges, radii, types)

    return Vasculature(point_data, segment_structure, chain_structure, chain_connectivity)
