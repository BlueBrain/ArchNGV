""" Synthesis entry function """

import os
import logging
# TODO: used by `eval` below; remove along with `eval`
import numpy as np  # pylint: disable=unused-import

import morphio
from tns.spatial.point_cloud import PointCloud  # pylint: disable=import-error

from archngv.utils.decorators import log_execution_time, log_start_end
from archngv.exceptions import NGVError

from archngv.building.morphology_synthesis.perimeters import add_perimeters_to_morphology

from .data_extraction import obtain_endfeet_data
from .data_extraction import obtain_synapse_data
from .data_extraction import obtain_cell_properties

from .tns_wrapper import TNSGrowerWrapper


L = logging.getLogger(__name__)


@log_start_end
@log_execution_time
def synthesize_astrocyte(astrocyte_index,
                         cell_data_path,
                         microdomains_path,
                         synaptic_data_path,
                         gliovascular_data_path,
                         gliovascular_connectivity_path,
                         neuroglial_conn_path,
                         tns_parameters_path,
                         tns_distributions_path,
                         morphology_directory,
                         endfeet_areas_path,
                         parameters):
    """ Synthesize the endfeet for one astrocyte
    """
    # pylint: disable=too-many-arguments

    # initialize tns grower adapter
    astro_grower = TNSGrowerWrapper(tns_parameters_path, tns_distributions_path)

    # astrocyte properties
    cell_name, soma_pos, soma_rad, microdomain = \
        obtain_cell_properties(astrocyte_index,
                               cell_data_path,
                               microdomains_path)

    # extract synapses coordinates
    synapses = obtain_synapse_data(astrocyte_index,
                                   synaptic_data_path,
                                   neuroglial_conn_path)

    if synapses is not None:
        # space colonization point cloud parameters
        radius_of_influence, removal_radius = parameters['point_cloud']

        # create point cloud
        point_cloud = PointCloud(synapses.values, radius_of_influence, removal_radius)
    else:

        point_cloud = None

    # register point cloud if exists
    astro_grower.add_point_cloud(point_cloud)

    # set soma position and radius
    astro_grower.set_soma_properties(soma_pos, soma_rad)

    # constrain the grower to its microdomain extent
    astro_grower.add_microdomain_boundary(microdomain)

    # get endfeet targets on vasculature surface
    endfeet_data = obtain_endfeet_data(astrocyte_index,
                                       gliovascular_data_path,
                                       gliovascular_connectivity_path,
                                       endfeet_areas_path)

    if endfeet_data is not None:

        # lambda function is passed from config as string
        lambda_string = parameters['attraction_field']

        # add the targets to grow to and the attraction field
        # function which depends on the distance to the target
        # TODO: consider using Equation instead of `eval`?
        field_function = eval(lambda_string)  # pylint: disable=eval-used
        astro_grower.set_endfeet_targets(endfeet_data.targets, field_function)

        # we need to make sure that our extracted homology will suffice
        # in order to reach the target. If they are shorter, we scale them.
        astro_grower.enable_endfeet_barcode_scaling()

        # determine the orientations of the primary processes by using
        # the geometry of the microdomain and its respective anisotropy
        astro_grower.set_process_orientations_from_microdomain(soma_pos, microdomain, endfeet_data.targets)

    else:

        astro_grower.remove_endfeet_properties()

        # determine the orientations of the primary processes by using
        # the geometry of the microdomain and its respective anisotropy
        astro_grower.set_process_orientations_from_microdomain(soma_pos, microdomain, None)

    astro_grower.grow()

    if parameters['perimeter_distribution']['enabled']:
        L.info('Distributing perimeters...')
        add_perimeters_to_morphology(astro_grower.morphology, parameters['perimeter_distribution'])

    morphology_output_file = os.path.join(morphology_directory, cell_name + '.h5')
    astro_grower.write(morphology_output_file)

    # TODO remove this when write neuron ordering is implemented
    morphology = morphio.Morphology(morphology_output_file, options=morphio.Option.nrn_order)
    morphology.as_mutable().write(morphology_output_file)

    _sanity_checks(cell_name, morphology)


def _sanity_checks(cell_name, morphology):
    """ Various checks ensuring morphology corectedness """

    for section in morphology.iter():

        if section.is_root:
            continue

        try:
            assert np.allclose(section.points[0], section.parent.points[-1])
        except AssertionError:
            msg = 'Morphology {} is missing duplicate points.\n'.format(cell_name)
            msg += '\t Section {}, Points: {}, Parent last point: {}'.format(
                section.id, section.points, section.parent.points[-1])
            raise NGVError(msg)

        try:
            assert len(section.points) > 1
        except AssertionError:
            msg = 'Morphology {} has one point sections.\n'.format(cell_name)
            msg += '\t Section {}, Points: {}, Parent points: {}'.format(
                section.id, section.points, section.parent.points[-1])
            raise NGVError(msg)
