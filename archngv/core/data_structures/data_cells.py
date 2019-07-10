"""
Data structures for basic cell characteristics
"""

import os

import numpy as np

from archngv.core.data_structures.common import H5ContextManager


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

        assert len(self.astrocyte_gids) == \
               len(self.astrocyte_names) == \
               len(self.astrocyte_radii) == \
               len(self.astrocyte_positions), 'hdf5 datasets are not of the same size'

    def __len__(self):
        "return cell data size"
        return len(self.astrocyte_positions)

    @property
    def astrocyte_point_data(self):
        """ Returns stacked astrocyte positions and radii
        """
        return np.column_stack((self.astrocyte_positions, self.astrocyte_radii))

    @property
    def n_cells(self):
        """ Number of cells """
        return self.__len__()


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
        """ Absolute path to the astrocyte morphology corresponding
        to the given index.
        """
        cell_filename = self.astrocyte_names[astrocyte_index] + '.h5'
        return os.path.join(self._config.morphology_directory, cell_filename)

    def morphology_object(self, astrocyte_index):
        """ Readonly morphology object using morphio
        Returns:
            A morphio read only object

        Notes:
            You need to pip install archngv[core] or archngv[all] to have access to this feature
        """
        from morphio import Morphology
        return Morphology(self.morphology_path(astrocyte_index))
