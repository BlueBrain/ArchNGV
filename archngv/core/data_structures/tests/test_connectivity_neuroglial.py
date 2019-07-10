import os

import numpy.testing as npt

from archngv.core.data_structures.connectivity_neuroglial import NeuroglialConnectivity


TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'neuroglial_connectivity.h5')


def test_all():
    with NeuroglialConnectivity(TEST_FILE) as conn:
        npt.assert_equal(
            conn.astrocyte_synapses(0),
            [11]
        )
        npt.assert_equal(
            conn.astrocyte_synapses(1),
            [11, 22, 33]
        )
