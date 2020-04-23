import os

import numpy.testing as npt

from archngv.core.connectivities import NeuroglialConnectivity


'''
import pandas as pd
from archngv.building.exporters import export_neuroglial_connectivity
from voxcell.sonata import NodePopulation

astrocyte_data = pd.DataFrame({'astrocyte_id': [0, 1, 1, 1],
                               'neuron_id': [2, 0, 0, 2],
                               'synapse_id': [11, 11, 22, 33],
                               })
neurons = NodePopulation('default', 2)
astrocytes = NodePopulation('default', 3)

export_neuroglial_connectivity(astrocyte_data, neurons, astrocytes, 'neuroglial_connectivity.h5')
'''

TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'neuroglial_connectivity.h5')


def test_all():
    with NeuroglialConnectivity(TEST_FILE) as conn:
        npt.assert_equal(conn.astrocyte_synapses(0), [11])
        npt.assert_equal(conn.astrocyte_synapses(1), [11, 22, 33])

        npt.assert_equal(conn.astrocyte_neurons(0), [2, ])
        npt.assert_equal(conn.astrocyte_neurons(1), [0, 2])
