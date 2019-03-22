import os
import bisect
import logging

from copy import deepcopy
from functools import partial

import scipy.stats
import numpy

import neurom
import tmd, tns
from morphspatial import collision
from morphio import PointLevel, SectionType
from tns.generate.grower import NeuronGrower

from .detail.morphology_types import INTERMEDIATE_ASTROCYTE_TYPES as ASTROCYTE_TYPES
from .detail.morphology_types import MAP_TO_NEURONAL
from .detail.synaptic_seeds import PointCloud
#from .projections import grow_endfeet_projections
from .detail.domain_orientation import orientation_function

from .detail.annotation import export_endfoot_location
from .detail.annotation import export_synapse_location

from archngv.core.data_structures.connectivity_neuroglial import NeuroglialConnectivity
from archngv.core.data_structures.connectivity_gliovascular import GliovascularConnectivity
from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
from archngv.core.data_structures.data_gliovascular import GliovascularData
from archngv.core.data_structures.data_synaptic import SynapticData


L = logging.getLogger(__name__)


SMALL_PROCESS   = MAP_TO_NEURONAL['small_process']
DOMAIN_PROCESS  = MAP_TO_NEURONAL['domain_process']
ENDFOOT_PROCESS = MAP_TO_NEURONAL['endfoot']


def orientation_from_target(soma_center, target):
    direction = target - soma_center
    return direction / numpy.linalg.norm(direction)


def get_closest_barcode(barcode_distribution, target_distance, return_max_radial=False, supremum=True):

    bar_lengths = lambda bar: numpy.abs(bar[0] - bar[1])
    get_max_radial = lambda barcode: max(map(bar_lengths, barcode))

    max_radials = list(map(get_max_radial, barcode_distribution))

    sorted_idx = sorted(range(len(max_radials)), key=lambda n: max_radials[n])
    sorted_radials = [max_radials[i] for i in sorted_idx]

    sorted_index = bisect.bisect_right(sorted_radials, target_distance) if supremum else \
                    bisect.bisect_left(sorted_radials, target_distance) - 1

    # in boundaries
    sorted_index = numpy.clip(sorted_index, 0, len(sorted_radials) - 1)

    # go back to the unsorted index
    index = sorted_idx[sorted_index]

    if return_max_radial:
        return barcode_distribution[index], max_radials[index]
    else:
        return barcode_distribution[index]


def astrocyte_data(config, astrocyte_index):
    """ Get the endfeet target points on the surface of the vasculature if any
    """

    # get connectivity with vasculature and the synapses
    with SynapticData(config.output_paths('synaptic_data')) as sn_data, \
         GliovascularData(config.output_paths('gliovascular_data')) as gv_data, \
         NeuroglialConnectivity(config.output_paths('neuroglial_connectivity')) as ng_conn, \
         GliovascularConnectivity(config.output_paths('gliovascular_connectivity')) as gv_conn, \
         MicrodomainTesselation(config.output_paths('overlapping_microdomain_structure')) as md_fd:

        endfeet_indices = gv_conn.astrocyte.to_endfoot(astrocyte_index)

        if len(endfeet_indices) > 0:
            endfeet = gv_data.endfoot_surface_coordinates[sorted(endfeet_indices)]
            L.info('Found {} for astrocyte index {}'.format(len(endfeet), astrocyte_index))
        else:
            endfeet = None
            L.info('No endfeet found for astrocyte index {}'.format(astrocyte_index))

        synapse_positions = sn_data.synapse_coordinates[sorted(ng_conn.astrocyte.to_synapse(astrocyte_index))]

        L.info('Found {} synapses for astrocyte index {}'.format(len(synapse_positions), astrocyte_index))

        domain = md_fd.domain_object(astrocyte_index)

    return endfeet, synapse_positions, domain


def _initialize_morphology_grower(config, soma_position, endfeet_targets, domain, synapse_positions):


    # convert the coordinates into normalized vectors form the soma
    endfeet_orientations = None if endfeet_targets is None else \
    [orientation_from_target(soma_position, target) for target in endfeet_targets]

    # number of primary processes from literature
    domain_process_num_trees = int(scipy.stats.norm(4.431391289748588, 0.3331223431693443).rvs())

    # convert to orientations taking into account
    # then endfeet ones
    domain_process_orientations, domain_process_lengths = \
    orientation_function(soma_position, domain, domain_process_num_trees,
                         predetermined_orientations=endfeet_orientations,
                         return_lengths=True,
                         face_interpolation=False)

    domain_process_targets = soma_position + domain_process_orientations * domain_process_lengths[:, numpy.newaxis]

    distributions = _get_synthesis_distributions(config, soma_position, domain_process_targets, endfeet_targets)

    parameters = _get_synthesis_parameters(soma_position, domain,
                                           endfeet_orientations, synapse_positions,
                                           domain_process_orientations, config)

    spatial = _get_spatial_data(domain, synapse_positions.copy(), endfeet_targets)


    L.info('Morphology Grower Started')

    # create a grower
    morphology_grower = NeuronGrower(input_distributions=distributions,
                                     input_parameters=parameters,
                                     spatial_data=spatial)

    return morphology_grower


def _get_synthesis_distributions(ngv_config, soma_center, domain_process_targets, endfeet_targets):

    L.info('Extracting synthesis distributions.')

    raw_morphology_directory = ngv_config.input_paths('synthesis_input_cells')

    # statistical distributions
    distr = tns.extract_input.distributions(raw_morphology_directory)

    # hack it externally instead of drawing from inside
    distr[DOMAIN_PROCESS]['num_trees'] = \
    {'norm': {'std': 0, 'mean': len(domain_process_targets)}}

    apical_persistence_diagram = distr['apical']['persistence_diagram']

    # closest radial trees
    distr[DOMAIN_PROCESS]['persistence_diagram'] = \
    [get_closest_barcode(apical_persistence_diagram, numpy.linalg.norm(soma_center - target), supremum=True) for target in domain_process_targets]

    if endfeet_targets is not None:

        if len(distr[ENDFOOT_PROCESS]['persistence_diagram']) == 0:

            distr[ENDFOOT_PROCESS] = deepcopy(distr[DOMAIN_PROCESS])
            L.info('No Persistence diagram found for domain endfoot process. Copying the domain one.')

        distr[ENDFOOT_PROCESS]['persistence_diagram'] = \
        [get_closest_barcode(apical_persistence_diagram, numpy.linalg.norm(soma_center - target), supremum=True) for target in endfeet_targets]

        # force number of endfeet that are predetermined
        distr[ENDFOOT_PROCESS]['num_trees'] = {'norm': {'mean': len(endfeet_targets), 'std': 0}}

    L.info('Extracting synthesis distributions completed.')

    return distr


def _get_synthesis_parameters(soma_center, domain, endfeet_orientations,
                              synapse_positions, domain_process_orientations,ngv_config):
    """ Setup parameters, distributions and spatial data for synthesis
    """

    L.info('Setting synthesis parameters.')

    # extract topology parameters
    parameters = \
    tns.extract_input.parameters(neurite_types=[SMALL_PROCESS, DOMAIN_PROCESS], method='tmd') if endfeet_orientations is None else \
    tns.extract_input.parameters(neurite_types=[SMALL_PROCESS, DOMAIN_PROCESS, ENDFOOT_PROCESS], method='tmd')

    # cell origin
    parameters['origin'] = soma_center

    # grow processes with respect to the seeds
    parameters[SMALL_PROCESS]['growth_method']   = 'space_colonization'
    parameters[DOMAIN_PROCESS]['growth_method']  = 'space_colonization'

    #################################################################################################

    parameters[DOMAIN_PROCESS]['orientation'] = domain_process_orientations.tolist()
    L.debug('Apical Orientations: {}'.format(parameters[DOMAIN_PROCESS]['orientation']))

    # assign synthesis randomness and targeting
    parameters[SMALL_PROCESS]['randomness'] = 0.2
    parameters[SMALL_PROCESS]['targeting']  = 0.5

    parameters[DOMAIN_PROCESS]['randomness'] = 0.1
    parameters[DOMAIN_PROCESS]['targeting']  = 0.9

    if endfeet_orientations is not None:

        # if there are no endfeet in the morphologies use the apical
        # topology instead
        if len(parameters[ENDFOOT_PROCESS]) == 0:

            parameters[ENDFOOT_PROCESS] = deepcopy(parameters[DOMAIN_PROCESS])
            parameters[ENDFOOT_PROCESS]['tree_type'] = 2
            L.info('Enfeet was not found. Using  domain process topology instead')

        # specific method for endfeet
        parameters[ENDFOOT_PROCESS]['growth_method'] = 'space_colonization_endfoot' 

        # given orientations
        parameters[ENDFOOT_PROCESS]['orientation'] = list(endfeet_orientations)
        L.debug('Endfeet Orientations: {}'.format(parameters[ENDFOOT_PROCESS]['orientation']))

        parameters[ENDFOOT_PROCESS]['randomness'] = 0.1
        parameters[ENDFOOT_PROCESS]['targeting']  = 1.0

    L.info('Setting synthesis parameters completed.')
    return parameters


def _get_spatial_data(domain, synapse_positions, endfeet_targets):

    L.info('Generating of spatial data.')

    # create spatial data
    spatial = {}

    # collision boundary
    spatial['collision_handler'] = partial(collision.convex_shape_with_point,
                                           domain.points[domain.triangles[:, 0]],
                                           domain.face_normals)

    # synaptic point cloud
    spatial['seeds'] = PointCloud(synapse_positions, 1., 1.)

    if endfeet_targets is not None:
        spatial[ENDFOOT_PROCESS] = {'endfeet_targets': endfeet_targets}

    L.info('Generating of spatial data completed.')

    return spatial


def _synthesize_astrocyte(args):
    """ Synthesis worker
    """
    index, soma_position, cell_name, config = args
    endfeet_targets, synapse_positions, domain = astrocyte_data(config, index)

    L.info('Synthesis of {} started.'.format(cell_name))

    morphology_grower = _initialize_morphology_grower(config, soma_position, endfeet_targets, domain, synapse_positions)
    morphology_grower.grow()
    #neuron.diametrize()

    # write morphology to file
    output_file = os.path.join(config.morphology_directory, cell_name + '.h5')
    morphology_grower.neuron.write(output_file)

    if endfeet_targets is not None:
        # reload morphology in immutable mode to extract indices
        export_endfoot_location(output_file, endfeet_targets)

    # get the location of the synapse in the morphology
    export_synapse_location(output_file, synapse_positions)

    L.info('Synthesis of {} completed.'.format(cell_name))


def synthesize_astrocytes(ngv_config, positions, cell_names, map_func):

    def data_generator(ngv_config, positions, cell_names):
        for index, (pos, name) in enumerate(zip(positions, cell_names)):
            yield index, pos, name, ngv_config

    map_func(_synthesize_astrocyte, data_generator(ngv_config, positions, cell_names))

