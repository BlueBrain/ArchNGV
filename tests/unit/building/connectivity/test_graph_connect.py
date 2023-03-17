import numpy as np
import pytest
import pandas as pd
import numpy.testing as npt
from pathlib import Path

from archngv.core.datasets import Microdomains
from archngv.building.connectivity.detail.gliovascular_generation.graph_reachout import strategy
from archngv.building.connectivity.detail.gliovascular_generation import graph_connect as tested

DATA_DIR = Path(__file__).resolve().parent / "../../app/data"
BUILD_DIR = DATA_DIR / "frozen-build"


def test_domains_to_vasculature():
    cell_ids = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    reachout_strategy_function = reachout_strategy_function = strategy("maximum_reachout")
    potential_targets = pd.DataFrame(
        {
            "x": [26.0, 26.0, 30.0, 33.0],
            "y": [23.0, 26.0, 30.0, 33.0],
            "z": [23.0, 26.0, 30.0, 33.0],
            "r": [2.96, 2.69, 2.65, 2.65],
            "edge_index": [0, 1, 1, 2],
            "vasculature_section_id": [0, 1, 1, 1],
            "vasculature_segment_id": [0, 0, 1, 1],
        }
    )

    properties = {"reachout_strategy": "maximum_reachout", "endfeet_distribution": [2, 2, 0, 15]}

    domains = Microdomains(BUILD_DIR / "microdomains.h5")

    astrocyte_target_edges = tested.domains_to_vasculature(
        cell_ids, reachout_strategy_function, potential_targets, domains, properties
    )

    print(f"astrocyte_target_edges {astrocyte_target_edges}")
    npt.assert_array_equal(astrocyte_target_edges, [[1, 1],  [1, 3],  [1, 0],  [3, 0],  [3, 1],  [3, 3]])
                                                    
