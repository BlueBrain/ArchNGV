""" H5 context manager """
import h5py


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
