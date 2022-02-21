import logging
import os
import tempfile

import numpy as np

from archngv.building.cell_placement.export import export_cell_placement_data

logging.basicConfig(level=logging.DEBUG)

L = logging.getLogger(__name__)

"""
class Payloads:

    n_astrocytes = 10
    radii = np.random.random(n_astrocytes)
    positions = np.random.random((n_astrocytes, 3))
    ids = np.arange(5, n_astrocytes + 5, dtype=np.uint64)


def initialize_basalt_graph(filepath, ids):
    from basalt.ngv import NGVGraph, VertexType, Astrocyte

    ngv_graph = NGVGraph(path=filepath)
    types = np.full(len(ids), fill_value=VertexType.ASTROCYTE.value, dtype=np.int32)

    basalt_payloads = [Astrocyte().serialize() for _ in range(len(ids))]

    ngv_graph.vertices.add(types, ids, payloads=basalt_payloads)
    ngv_graph.commit()


def test_basalt_exporter():

    from basalt.ngv import NGVGraph

    payloads = Payloads()

    with tempfile.TemporaryDirectory() as dirpath:

        filepath = os.path.join(dirpath, 'basalt_network.db')

        # generate db with astrocytes and ids only
        initialize_basalt_graph(filepath, payloads.ids)

        # export cell placement payloads
        export_cell_placement_data(filepath,
                                   payloads.ids,
                                   payloads.positions,
                                   payloads.radii,
                                   method='basalt')

        # load the astrocytes and test
        ngv_graph = NGVGraph(path=filepath)
        astrocytes = ngv_graph.astrocytes

        for i, astrocyte_id in enumerate(payloads.ids):

            payload = astrocytes[astrocyte_id].data
            expected_pos = payloads.positions[i]

            assert np.allclose(expected_pos[0], payload.soma_center.x)
            assert np.allclose(expected_pos[1], payload.soma_center.y)
            assert np.allclose(expected_pos[2], payload.soma_center.z)
            assert np.isclose(payloads.radii[i], payload.soma_radius)

"""
