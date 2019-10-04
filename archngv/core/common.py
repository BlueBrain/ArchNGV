""" Commonly used context managers. """

import h5py
import libsonata

from archngv.exceptions import NGVError


def _open_population(h5_filepath):
    storage = libsonata.EdgeStorage(h5_filepath)
    populations = storage.population_names
    if len(populations) != 1:
        raise NGVError(
            "Only single-population edge collections are supported (found: %d)" % len(populations)
        )
    return storage.open_population(list(populations)[0])


class H5ContextManager(object):
    """ Context manager for hdf5 files """

    def __init__(self, filepath):
        self._fd = h5py.File(filepath, 'r')

    def close(self):
        """ Close hdf5 file """
        self._fd.close()

    def __enter__(self):
        """ Context mananger entry """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ And exit """
        self.close()


class EdgesContextManager(object):
    """ Context manager for accessing SONATA Edges """
    def __init__(self, filepath):
        self.filepath = filepath
        self._impl = _open_population(self.filepath)  # pylint: disable=attribute-defined-outside-init

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._impl
