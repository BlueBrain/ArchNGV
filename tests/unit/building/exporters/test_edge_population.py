import shutil
import tempfile
from pathlib import Path

import pytest
from archngv.core.datasets import GliovascularConnectivity
import numpy as np
from numpy import testing as npt

from archngv.exceptions import NGVError
from archngv.building.exporters import edge_populations as tested

DATA = Path(__file__).resolve().parent / 'data'


@pytest.fixture(scope='function')
def edge_population_file():

    with tempfile.NamedTemporaryFile(suffix='.h5') as tfile:

        shutil.copy(str(DATA / 'gliovascular.h5'), tfile.name)
        yield tfile.name


@pytest.fixture(scope='function')
def properties(edge_population_file):
    gv_conn = GliovascularConnectivity(edge_population_file)
    return {name: gv_conn.get_properties(name) for name in gv_conn.property_names}


def test_add_properties_to_edge_population(edge_population_file, properties):

    new_properties = {
        'john': np.arange(23),
        'paul': np.arange(1, 24)
    }

    tested.add_properties_to_edge_population(
        edge_population_file, 'gliovascular', new_properties)

    gv_conn = GliovascularConnectivity(edge_population_file)

    npt.assert_array_equal(gv_conn.property_names, set(properties.keys()) | {'john', 'paul'})

    props = gv_conn.get_properties(['john', 'paul'])
    npt.assert_array_equal(props[:, 0], np.arange(23))
    npt.assert_array_equal(props[:, 1], np.arange(1, 24))

    for prop_name, prop_values in properties.items():
        npt.assert_allclose(prop_values, gv_conn.get_properties(prop_name))


def test_add_properties_to_edge_population__assertions(edge_population_file, properties):

    new_properties = {
        'john': np.arange(23),
        'paul': np.arange(5)
    }

    # wrong length
    with pytest.raises(NGVError):
        tested.add_properties_to_edge_population(edge_population_file, 'gliovascular', new_properties)

    new_properties = {
        'john': np.arange(23),
    }

    # property exists
    with pytest.raises(NGVError):
        tested.add_properties_to_edge_population(edge_population_file, 'gliovascular', new_properties)
