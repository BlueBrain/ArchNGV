""" Wrapper of TNS AstrocyteGrower. It takes care of all
the functionality required for astrocyte synthesis.
"""

import json
import logging
import numpy as np
import scipy.stats

from tns import AstrocyteGrower
from tns.morphmath import sample
from tns.morphmath.field import PointTarget
from tns.morphmath.field import PointAttractionField

from morphspatial.collision import convex_shape_with_point

from .ph_modification import scale_barcode 
from .synaptic_seeds import PointCloud
from .domain_orientation import orientations_from_domain


L = logging.getLogger(__name__)


ENDFOOT_TYPE = 'axon'
PROCESS_TYPE = 'basal'


class TNSGrowerWrapper(object):
    """ Adapter Class for tns AstrocyteGrower

    Args:
        parameters_path: string
            Absolute path to tns parameters json file.
        distributions_path: string
            Absolute path to tns distributions json file.

    Attrs:
        parameters: dict
            TNS parameters dict.
        distributions: dict
            TNS distributions dict.
        context: dict
            TNS context dict.
        morphology: MorphIO.Morphology
    """
    def __init__(self, parameters_path, distributions_path):

        with open(parameters_path, 'r') as parameters_fd, \
             open(distributions_path, 'r') as distributions_fd:

            self._parameters = json.load(parameters_fd)
            self._distributions = json.load(distributions_fd)

        self._morphology = None
        self._context = {'collision_handle': lambda _: False}

    def add_collision_handle(self, collision_handle):
        """ Assigns as a collision object a function which takes as an
        input a point and returns True if there is a collision, otherwise False

        Example: collision_handle = lambda _: False (No collision takes place ever)
        """
        self._context['collision_handle'] = collision_handle

    def set_soma_properties(self, soma_position, soma_radius):
        """ Set soma positions and radius in the TNS parameters.

        Args:
            soma_position: array[float, (3,)]
            soma_radius: float

        Due to the fact that TNS handles distributions for the soma radii, but
        we have them precalculated during placement, we set a normal with std 0.0
        and mean of the radius we want.
        """
        self._parameters['origin'] = np.asarray(soma_position, dtype=np.float)
        self._distributions['soma']['size'] = {'norm': {'mean': soma_radius, 'std': 0.0}}

    def add_microdomain_boundary(self, microdomain):
        """ Assigns the microdomain boundary as a collision bounding hull
        that will contain the grower.
        """
        # one point per face that belongs to that plane
        face_points = microdomain.face_points
        face_normals = microdomain.face_normals

        collision_handle = \
            lambda point: not convex_shape_with_point(face_points, face_normals, point)

        self.add_collision_handle(collision_handle)

    def set_process_orientations_from_microdomain(self, soma_center,
                                                      microdomain,
                                                      endfeet_targets):
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
                                                   microdomain,
                                                   n_trunks,
                                                   fixed_targets=endfeet_targets)
        self._parameters[PROCESS_TYPE]['orientation'] = orientations

    def set_endfeet_targets(self, target_points, field_function):
        """ Set targets for the processes to grow to.
        """
        target_objects = [PointTarget(point) for point in target_points]
        self._parameters[ENDFOOT_TYPE]['targets'] = target_objects

        self._context['field'] = PointAttractionField(field_function)

    def add_point_cloud(self, coordinates, radius_of_influence, removal_radius):
        """ Add point cloud in synthesis context
        """
        point_cloud = PointCloud(coordinates, radius_of_influence, removal_radius)
        self._context['point_cloud'] = point_cloud

    def remove_endfeet_properties(self):
        """ If no endfeet targets are available, make sure that
        tns will not grow trees for the endfoot type
        """
        self._parameters['grow_types'].remove(ENDFOOT_TYPE)

    def enable_endfeet_barcode_scaling(self):
        '''Modifies the input parameters to match the input data
           taken from the spatial properties of the Atlas:
           The reference_thicness is the expected thickness of input data
           The target_thickness is the expected thickness that the synthesized
           cells should live in. Input should be modified accordingly
        '''
        self._parameters[ENDFOOT_TYPE].update({'modify_target': {'funct': scale_barcode,
                                               'kwargs': {}}})

    def grow(self):
        """ Run morphology synthesis
        """
        tns_grower = AstrocyteGrower(input_distributions=self._distributions,
                                     input_parameters=self._parameters,
                                     context=self._context)
        self._morphology = tns_grower.grow()

    def write(self, filepath):
        """ Write morphology to file
        """
        assert self._morphology is not None
        self._morphology.write(filepath)
