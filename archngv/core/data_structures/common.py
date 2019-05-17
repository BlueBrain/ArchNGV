import h5py


class H5ContextManager(object):

    def __init__(self, filepath):
        self._fd = h5py.File(filepath, 'r')

    def close(self):
        self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
