import os

import numpy.testing as npt

from archngv.core.data_synaptic import SynapticData


TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'synaptic_data.h5')


def test_all():
    with SynapticData(TEST_FILE) as syn_data:
        assert syn_data.n_synapses == 4
        assert syn_data.n_neurons == 2
        npt.assert_equal(
            syn_data.afferent_gids(),
            [0, 1, 1, 1]
        )
        npt.assert_equal(
            syn_data.afferent_gids([0, 2]),
            [0, 1]
        )
        npt.assert_almost_equal(
            syn_data.synapse_coordinates(),
            [
                [2110., 2120., 2130.],
                [2111., 2121., 2131.],
                [2112., 2122., 2132.],
                [2113., 2123., 2133.],
            ]
        )
        npt.assert_almost_equal(
            syn_data.synapse_coordinates([0, 2]),
            [
                [2110., 2120., 2130.],
                [2112., 2122., 2132.],
            ]
        )
