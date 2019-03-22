import os
import h5py
import logging

import numpy as np

from .common import H5ContextManager


L = logging.getLogger(__name__)


class CellData(H5ContextManager):
    """ Data structure for the collection of cell characteristics. Only the actual
    file is required for accessing the respetive data with this class. No relative path
    data is available from this entry point.
    """
    def __init__(self, filepath):
        super(CellData, self).__init__(filepath)

        self.astrocyte_positions = self._fd['/positions']
        self.astrocyte_radii = self._fd['/radii']

        self.astrocyte_gids = self._fd['/ids']
        self.astrocyte_names = self._fd['/names']

    @property
    def astrocyte_point_data(self):
        return np.column_stack((self.astrocyte_positions, self.astrocyte_radii))

    @property
    def n_cells(self):
        return len(self.astrocyte_positions)


class CellDataInfo(CellData):
    """ Rich circuit information after it has been created. This requires an
    ngv_config with more info than just the path to cell_data. This way access to
    morphologies is also possible.
    """
    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('cell_data')
        super(CellDataInfo, self).__init__(filepath)
        self._config = ngv_config

    def morphology_path(self, astrocyte_index):
        cell_name = self.astrocyte_names[astrocyte_index]
        return os.path.join(self.config.morphology_directory, '{}.h5'.format(cell_name))

    def morphology_object(self, astrocyte_index):
        import morphio
        return morphio.Morphology(self.morphology_path(astrocyte_index))


