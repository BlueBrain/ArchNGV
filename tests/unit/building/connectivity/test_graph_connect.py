from mock import patch
import numpy as np
import numpy.testing as npt
import pandas as pd
from pathlib import Path

from archngv.core.datasets import Microdomains
from archngv.building.connectivity.detail.gliovascular_generation.graph_reachout import strategy
from archngv.building.connectivity.detail.gliovascular_generation import graph_connect as tested

DATA_DIR = Path(__file__).resolve().parent / "../../app/data"
BUILD_DIR = DATA_DIR / "frozen-build"

@patch('scipy.stats._distn_infrastructure.rv_continuous_frozen.rvs')
def test_domains_to_vasculature(mocked_function):
    mocked_function.return_value = np.array([2, 1] )
    cell_ids = np.array([0, 1 ], dtype=np.int16)
    reachout_strategy_function = reachout_strategy_function = strategy("maximum_reachout")

    potential_targets = pd.DataFrame(
        {
            "x": [0.0 , 0.0 , 40.0],
            "y": [20.0, 40.0, 40.0],
            "z": [40.0, 60.0, 30.0],
            "r": [0., 0., 0.],
            "edge_index": [0, 1, 1],
            "vasculature_section_id": [0, 1, 1],
            "vasculature_segment_id": [0, 0, 1],
        }
    )
    
    properties = {"reachout_strategy": "maximum_reachout", "endfeet_distribution": [2, 2, 0, 15]}
    domains = Microdomains(BUILD_DIR / "microdomains.h5")
        
    """ This is the omain Bounding Box values
    (-7.6956673, 33.304604, 20.442867, 77.779236, 77.40124, 77.37364)
    (-7.914425, 7.0291786, -7.3803453, 77.56047, 77.630165, 47.34516)
    """

    astrocyte_target_edges = tested.domains_to_vasculature(
        cell_ids, reachout_strategy_function, potential_targets, domains, properties
    )

    print(f"astrocyte_target_edges {astrocyte_target_edges}")
    npt.assert_array_equal(astrocyte_target_edges, [[0, 1], [1, 2]])

