import time
import logging
import numpy

from pandas import DataFrame

from voxcell import CellCollection

from morphmath import rand_rotation_matrix

from archngv import NGVConfig
from archngv.core.cell_placement.density import read_densities_from_file
from archngv.core.cell_placement.density import create_density_from_laminar_densities

L = logging.getLogger(__name__)

"""
def _translate_vasculature_to_atlas_space(vasculature, voxel_data):

    d_ll, d_uu = voxel_data.bbox
    v_ll, v_uu = vasculature.bounding_box.ranges

    pia_base_center = numpy.array([d_ll[0] + 0.5 * (d_uu[0] - d_ll[0]),
                                   d_uu[1],
                                   d_ll[2] + 0.5 * (d_uu[2] - d_ll[2])])

    vsc_base_center = numpy.array([v_ll[0] + 0.5 * (v_uu[0] - v_ll[0]),
                                   v_uu[1],
                                   v_ll[2] + 0.5 * (v_uu[2] - v_ll[2])])

    vasculature.points = vasculature.points - vsc_base_center + pia_base_center

def load_vasculature(ngv_config, translate_to_voxeldata):
    from archngv.vasculature_morphology import Vasculature

    vasculature_path = ngv_config.input_paths('vasculature')
    L.info('Loading Vasculature Dataset at %s', vasculature_path)

    vasculature = Vasculature.load(vasculature_path)

    if translate_to_voxeldata is not None:
        _translate_vasculature_to_atlas_space(vasculature, translate_to_voxeldata)

    return vasculature
"""

def create_repeat_project(original_config_path, repeat_number):

        with open(original_config_path, 'r') as fp:

            config_dict = json.load(fp)

        experiment_basename = config_dict['experiment_name']
        L.info("Base experiment name: {}".format(experiment_basename))
        new_experiment_name = "{}_repeat_{:05d}".format(experiment_basename, repeat_number)

        L.info("New experiment name: {}".format(new_experiment_name))

        config_dict['experiment_name'] = new_experiment_name

        return NGVConfig(config_dict, 'ngv_config')


def create_config(args):

    config_path = args.config

    config = \
    create_repeat_project(config_path, args.repeat) if args.repeat else NGVConfig.from_file(config_path)

    if args.seed:

        L.info("Seed provided: {}".format(args.seed))
        seed = args.seed

    else:

        
        seed = int(str(time.time()).replace('.', '')) % 4294967295
        L.info("Seed not provided. Generated: {}".format(seed))

    config._config['seed'] = seed

    L.info("Circuit Generation for experiment {} has started.".format(config.experiment_name))

    config.create_directories()
    config.save()
    config.generate_circuit_config()

    return config


def create_astrocyte_properties(cell_names):

    n_astrocytes = len(cell_names)

    properties = {
                'names'        : cell_names,
                'etype'        : ['NA' for _ in range(n_astrocytes)],
                'mtype'        : ['ASTROCYTE' for _ in range(n_astrocytes)],
                'morph_class'  : ['ASR' for _ in range(n_astrocytes)],
                'synapse_class': ['NA' for _ in range(n_astrocytes)]
                }

    return properties


def create_identity_orientations(n_astrocytes):

    orientations = numpy.zeros((n_astrocytes, 3, 3))
    orientations[:, 0, 0] = 1.
    orientations[:, 1, 1] = 1.
    orientations[:, 2, 2] = 1.

    return orientations


def create_astrocyte_collection(config, cell_names, positions, radii):

    L.info("Creating empty astrocyte morphologies from positions and radii.")

    assert len(cell_names) == len(positions) == len(radii)

    astro_collection = CellCollection()
    astro_collection.positions = positions

    n_astrocytes = len(positions)
    properties = create_astrocyte_properties(cell_names)

    if config._config['use_existing_astrocyte_morphologies']:

        orientations = numpy.dstack([rand_rotation_matrix() for _ in range(n_astrocytes)]).T
        L.info("Added random orientations.")

    else:

        orientations = helpers.create_identity_orientations()
        L.info("Added identity orientations.")

    astro_collection.orientations = orientations
    properties.update({'radius': radii})

    properties_dataframe = DataFrame.from_dict(properties)

    #astro_collection.create_empty_morphologies(positions, orientations, radii, properties, Astrocyte)
    astro_collection.add_properties(properties_dataframe)

    return astro_collection


def create_cell_placement_density(bounding_box, params):

    laminar_densities_filename = params['laminar_densities']

    densities, bins = read_densities_from_file(laminar_densities_filename)

    density = create_density_from_laminar_densities(1e-9 * densities, bins,
                                                    bounding_box, 1, 1)

    return density
