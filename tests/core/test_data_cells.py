import os
import pytest
import numpy as np


from archngv.core.data_cells import CellData
from archngv.building.exporters.export_ngv_data import export_cell_placement_data


N_CELLS = 11


class MockCellData(object):

    astrocyte_names = np.array(['cell_{}'.format(i) for i in range(N_CELLS)], dtype=bytes)

    astrocyte_gids = np.arange(42, N_CELLS + 42, dtype=np.uintp)

    astrocyte_positions = np.random.random((N_CELLS, 3))

    astrocyte_radii = np.random.random(N_CELLS)


@pytest.fixture(scope='session')
def cell_data_path(tmpdir_factory):

    #directory_path = tmpdir_factory.mktemp('files')

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'cell_data.h5')

    print(path)
    return path


@pytest.fixture(scope='module')
def mock_data(cell_data_path):

    data = MockCellData()

    export_cell_placement_data(cell_data_path,
                               data.astrocyte_gids,
                               data.astrocyte_names,
                               data.astrocyte_positions,
                               data.astrocyte_radii
    )

    return data


@pytest.fixture(scope='module')
def cell_data(cell_data_path, mock_data): # ensure  mock_data fixture is created first
    return CellData(cell_data_path)


def test_astrocyte_positions(mock_data, cell_data):

    assert np.allclose(mock_data.astrocyte_positions,
                       cell_data.astrocyte_positions)

def test_astrocyte_radii(mock_data, cell_data):

    assert np.allclose(mock_data.astrocyte_radii,
                       cell_data.astrocyte_radii)


def test_astrocyte_gids(mock_data, cell_data):
    assert np.all(mock_data.astrocyte_gids == cell_data.astrocyte_gids)


def test_astrocyte_names(mock_data, cell_data):

    assert len(mock_data.astrocyte_names) == len(cell_data.astrocyte_names)

    assert np.all(mock_data.astrocyte_names == cell_data.astrocyte_names)

