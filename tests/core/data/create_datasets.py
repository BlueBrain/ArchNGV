from pathlib import Path

import numpy as np
import pandas as pd

import voxcell

from archngv.building.exporters.edge_populations import _write_edge_population

TEST_DATA_DIR = Path(__file__).parent.resolve()


def get_data(filename):
    return str(Path(TEST_DATA_DIR / filename))


def create_neurons():
    cells = voxcell.CellCollection(population_name='default')
    cells.positions = np.array([[1., 2., 3.], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)

    props = {"layer": [1, 2, 3],
             "morphology": ["morph-A", "morph-B", "morph-C"],
             "mtype": ['L2_X', 'L6_Y', 'L6_Y'],
             "model_type": ["biophysical", "biophysical", "biophysical"],
             'model_template': ['hoc:small_bio', 'hoc:small_bio', 'hoc:small_bio']
             }
    cells.properties = pd.DataFrame(data=props)
    cells.save_sonata(get_data('nodes.h5'))


def create_vasculature():
    cells = voxcell.CellCollection(population_name='vasculature')

    props = {'type': [0, 0, 0],
             'section_id': [0, 0, 1],
             'segment_id': [0, 1, 2],
             'start_node_id': [0, 1, 2],
             'end_node_id': [1, 2, 3],
             'start_diameter': [0.5, 0.6, 0.7],
             'end_diameter': [0.51, 0.61, 0.71],
             }

    cells.properties = pd.DataFrame(data=props)

    cells.properties[['start_x', 'start_y', 'start_z']] = np.array([[1, 2, 3], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)
    cells.properties[['end_x', 'end_y', 'end_z']] = np.array([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]], dtype=np.float32)
    cells.save_sonata(get_data('vasculature_sonata.h5'))


def create_glia():
    cells = voxcell.CellCollection(population_name='astrocytes')
    cells.positions = np.array([[1., 2., 3.], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)

    props = {"morphology": ["morph-A", "morph-B", "morph-C"],
             "mtype": ['ASTROCYTE', 'ASTROCYTE', 'ASTROCYTE'],
             "model_type": ["biophysical", "biophysical", "biophysical"],
             'model_template': ['hoc:astro', 'hoc:astro', 'hoc:astro'],
             'radius': [1., 2., 3.]
             }
    cells.properties = pd.DataFrame(data=props)
    cells.save_sonata(get_data('glia.h5'))


def create_neuronal():

    props = {'afferent_center_x': [1110.0, 1111.0, 1112.0, 1113.0],
             'afferent_center_y': [1120.0, 1121.0, 1122.0, 1123.0],
             'afferent_center_z': [1130.0, 1131.0, 1132.0, 1133.0],
             'efferent_section_pos': [0.0, 0.0, 0.0, 0.0],
             'syn_weight': [1.0, 1.0, 1.0, 1.0]}

    _write_edge_population(output_path=get_data('edges.h5'),
                           source_population_name='default',
                           target_population_name='default',
                           source_population_size=3,
                           target_population_size=3,
                           source_node_ids=[2, 0, 0, 2],
                           target_node_ids=[0, 1, 1, 1],
                           edge_population_name='default',
                           edge_properties=props)


def create_neuroglial():

    props = {'synapse_id': [1, 3, 0, 1],
             'efferent_section_id': [0, 0, 1, 1],
             'efferent_segment_id': [0, 1, 0, 0],
             'efferent_segment_offset': [0.1, 0.2, 0.3, 0.4],
             }

    _write_edge_population(output_path=get_data('neuroglial.h5'),
                           source_population_name='astrocytes',
                           target_population_name='default',
                           source_population_size=3,
                           target_population_size=3,
                           source_node_ids=[0, 1, 2, 2],
                           target_node_ids=[1, 1, 0, 1],
                           edge_population_name='neuroglial',
                           edge_properties=props)


def create_gliovascular():

    props = {'endfoot_id': [0, 1, 2],
             'efferent_section_id': [0, 0, 1],
             'efferent_segment_id': [0, 1, 2],
             'endfoot_surface_x': [0.11, 0.21, 0.31],
             'endfoot_surface_y': [0.12, 0.22, 0.32],
             'endfoot_surface_z': [0.13, 0.23, 0.33],
             }

    _write_edge_population(output_path=get_data('gliovascular.h5'),
                           source_population_name='vasculature',
                           target_population_name='astrocytes',
                           source_population_size=3,
                           target_population_size=3,
                           source_node_ids=[0, 1, 2],
                           target_node_ids=[2, 1, 0],
                           edge_population_name='gliovascular',
                           edge_properties=props)


def create_glialglial():

    props = {'connection_id': [0, 1, 2, 3]}

    _write_edge_population(output_path=get_data('glialglial.h5'),
                           source_population_name='astrocytes',
                           target_population_name='astrocytes',
                           source_population_size=3,
                           target_population_size=3,
                           source_node_ids=[2, 0, 0, 2],
                           target_node_ids=[0, 1, 1, 1],
                           edge_population_name='glialglial',
                           edge_properties=props)


if __name__ == "__main__":
    create_neurons()
    create_vasculature()
    create_glia()
    create_neuronal()
    create_neuroglial()
    create_gliovascular()
    create_glialglial()
