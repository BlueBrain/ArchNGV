import os
import pandas as pd
import numpy.testing as npt
from voxcell.sonata import NodePopulation
from archngv.core.connectivities import GlialglialConnectivity
from archngv.building.exporters import export_glialglial_connectivity


TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'glialglial_connectivity.h5')

'''
N_ASTROCYTES = 3

astrocyte_data = pd.DataFrame({'astrocyte_target_id': [0, 0, 1, 1, 1, 2, 2],
                               'astrocyte_source_id': [2, 1, 0, 0, 2, 0, 1],
                               'connection_id': [11, 11, 22, 23, 33, 44, 55],
                               })


export_glialglial_connectivity(astrocyte_data, N_ASTROCYTES, TEST_FILE)
'''


def test_all():
    with GlialglialConnectivity(TEST_FILE) as conn:
        npt.assert_equal(conn.astrocyte_astrocytes(0), [1, 1, 2])
        npt.assert_equal(conn.astrocyte_astrocytes(1), [0, 2])
        npt.assert_equal(conn.astrocyte_astrocytes(2), [0, 1])

