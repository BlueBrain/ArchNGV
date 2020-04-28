""" Wrapper of TNS AstrocyteGrower. It takes care of all
the functionality required for astrocyte synthesis.
"""
import json
import logging
import numpy as np
import scipy.stats

from tns.morphmath.field import PointTarget  # pylint: disable=import-error
from tns.morphmath.field import PointAttractionField  # pylint: disable=import-error

from archngv.building.types import ASTROCYTE_TO_NEURON

from archngv.building.morphology_synthesis.domain_boundary import StopAtConvexBoundary
from archngv.building.morphology_synthesis.domain_orientation import orientations_from_domain


L = logging.getLogger(__name__)


ENDFOOT_TYPE = ASTROCYTE_TO_NEURON['endfoot']
PROCESS_TYPE = ASTROCYTE_TO_NEURON['domain_process']


L.info('Map: %s -> %s', 'endfoot', ENDFOOT_TYPE)
L.info('Map: %s -> %s', 'domain_process', PROCESS_TYPE)


def _default_context():
    return {'collision_handle': lambda _: None}


def _set_soma_properties(parameters, distributions, soma_position, soma_radius):
    """ Set soma positions and radius in the TNS parameters.

    Args:
        soma_position: array[float, (3,)]
        soma_radius: float

    Due to the fact that TNS handles distributions for the soma radii, but
    we have them precalculated during placement, we set a normal with std 0.0
    and mean of the radius we want.
    """
    parameters['origin'] = np.asarray(soma_position, dtype=np.float32)
    distributions['soma']['size'] = {'norm': {'mean': soma_radius, 'std': 0.0}}


def _assign_microdomain_as_collision_boundary(context, microdomain):
    """ Assigns the microdomain boundary as a collision bounding hull
    that will contain the grower.
    """
    context['collision_handle'] = StopAtConvexBoundary(
        microdomain.points, microdomain.triangles, microdomain.face_normals)


def _set_space_colonization_algorithm(parameters):
    """ Set growth method to space colonization """
    parameters[PROCESS_TYPE]['growth_method'] = 'tmd_space_colonization'
    parameters[ENDFOOT_TYPE]['growth_method'] = 'tmd_space_colonization_target'


def _set_fallback_algorithm(parameters):
    """ Fallback if there is no point cloud """
    parameters[PROCESS_TYPE]['growth_method'] = 'tmd'
    parameters[ENDFOOT_TYPE]['growth_method'] = 'tmd_target'


def _add_point_cloud(context, point_cloud):
    """ Add the point cloud to the context """
    context['point_cloud'] = point_cloud


def _add_endfeet_attractors(parameters, context, target_points, field_function):
    """ Set targets for the processes to grow to and be attracted by """
    target_objects = [PointTarget(point) for point in target_points]

    parameters[ENDFOOT_TYPE]['targets'] = target_objects
    context['field'] = PointAttractionField(field_function)


def _set_endfeet_barcode_scaling(parameters):
    """ Add modification function for scaling the barcodes """
    from archngv.building.morphology_synthesis.ph_modification import scale_barcode
    parameters[ENDFOOT_TYPE].update({'modify_target': {'funct': scale_barcode, 'kwargs': {}}})


def _remove_endfeet_properties(parameters):
    """ Remove the endfoot type from the growing types """
    parameters['grow_types'].remove(ENDFOOT_TYPE)


def _set_process_orientations_from_microdomain(parameters, soma_center, microdomain, endfeet_targets):
    """ Given the microdomain geometry and the endfeet targets, calculate the
    orientation of the primary processes from that geometry without overlapping
    with the endfeet targets.

    Args:
        soma_center: array[float, 3]
        microdomain: ConvexPolygon
        endfeet_targets: array[float, (N, 3)]
    """
    n_trunks = int(scipy.stats.norm(4.431391289748588, 0.3331223431693443).rvs())

    # n_trunks = sample.n_neurites(self._distributions[PROCESS_TYPE]['num_trees'])
    orientations, _ = orientations_from_domain(soma_center,
                                               microdomain.points,
                                               microdomain.triangles,
                                               n_trunks,
                                               fixed_targets=endfeet_targets)

    parameters[PROCESS_TYPE]['orientation'] = orientations


# pylint: disable=too-many-arguments
def create_tns_inputs(default_parameters_path,
                      default_distributions_path,
                      soma_position,
                      soma_radius,
                      microdomain=None,
                      point_cloud=None,
                      endfeet_data=None,
                      field_function=None,
                      endfeet_barcode_scaling=False):
    """ Generate inputs for tns astrocyte grower
    """
    with open(default_parameters_path, 'r') as parameters_fd, \
         open(default_distributions_path, 'r') as distributions_fd:

        parameters = json.load(parameters_fd)
        distributions = json.load(distributions_fd)

    # set origin and radius of cell soma
    _set_soma_properties(parameters, distributions, soma_position, soma_radius)

    context = _default_context()

    if microdomain is not None:
        _assign_microdomain_as_collision_boundary(context, microdomain)
    else:
        L.warning('No microdomain boundary provided.')

    if point_cloud is None:
        L.warning('No point cloud provided. Switched to regular synthesis')
        _set_fallback_algorithm(parameters)
    else:
        _add_point_cloud(context, point_cloud)
        _set_space_colonization_algorithm(parameters)

    if endfeet_data is not None:
        endfeet_targets = endfeet_data.targets
        _add_endfeet_attractors(parameters, context, endfeet_targets, field_function)

        # we need to make sure that our extracted homology will suffice
        # in order to reach the target. If they are shorter, we scale them.
        if endfeet_barcode_scaling:
            _set_endfeet_barcode_scaling(parameters)

    else:

        endfeet_targets = None
        _remove_endfeet_properties(parameters)

    # determine the orientations of the primary processes by using
    # the geometry of the microdomain and its respective anisotropy
    _set_process_orientations_from_microdomain(parameters, soma_position, microdomain, endfeet_targets)

    return parameters, distributions, context
