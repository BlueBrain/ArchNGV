from tns import extract_input, NeuronGrower

from .ph_modification import scale_barcode 
from .synaptic_seeds import PointCloud

import logging
import numpy as np

L = logging.getLogger(__name__)


ENDFOOT_TYPE = 'axon'


def endfeet_parameters(endfoot_synthesis_parameters):
    """ Get tns paramteters for endfeet
    """
    parameters = extract_input.parameters(neurite_types=[ENDFOOT_TYPE], method='tmd')

    parameters[ENDFOOT_TYPE]['growth_method'] = 'tmd_target_path'
    parameters[ENDFOOT_TYPE]['branching_method'] = 'bio_oriented'

    parameters[ENDFOOT_TYPE]['targeting'] = 0.01
    parameters[ENDFOOT_TYPE]['randomness'] = 0.32

    parameters[ENDFOOT_TYPE]['bias'] = 0.2
    parameters[ENDFOOT_TYPE]['bias_length'] = 10.

    parameters[ENDFOOT_TYPE]['collision_handle'] = lambda x: False

    parameters[ENDFOOT_TYPE]['influence_function'] = \
        lambda dmax, dmin, d: 0.003971651107390942 * d + 0.16091292712235578

    return parameters


class TNSGrowerWrapper(object):
    """ General Adapter Class for tns NeuronGrower
    """

    def __init__(self, input_distributions, input_parameters):

        self._distributions = input_distributions
        self._parameters = input_parameters

        self._morphology = None

    def add_collision_handle(self, collision_handle):
        """ Assigns as a collision object a function which takes as an
        input a point and returns True if there is a collision, otherwise False
        """
        self._parameters[ENDFOOT_TYPE]['collision_handle']

    def set_soma_properties(self, soma_position, soma_radius):
        """ Assigns the soma position and radius
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
            lambda point: not collision.convex_shape_with_point(face_points, face_normals, point)

        self.add_collision_handle(collision_handle)

    def grow(self):
        tns_grower = NeuronGrower(input_distributions=self._distributions,
                                  input_parameters=self._parameters,
                                  context=None)
        self._morphology = tns_grower.grow()

    def write(self, filepath):
        self._morphology.write(filepath)


class TNSEndfeetGrower(TNSGrowerWrapper):
    """ Adapter class for TNS NeuronGrower of targeting neurites
    """

    def __init__(self,
                 input_distributions,
                 endfeet_synthesis_params):

        self._parameters = endfeet_parameters(endfeet_synthesis_params)

        super(TNSEndfeetGrower, self).__init__(input_distributions, self._parameters)

    def set_targets(self, target_points):
        """ Set endfeet target destination, i.e. the points on the surface of
        the vasculature.
        """
        self._parameters[ENDFOOT_TYPE]['targets'] = np.asarray(target_points, dtype=np.float)


    def enable_barcode_scaling(self):
        '''Modifies the input parameters to match the input data
           taken from the spatial properties of the Atlas:
           The reference_thicness is the expected thickness of input data
           The target_thickness is the expected thickness that the synthesized
           cells should live in. Input should be modified accordingly
        '''
        self._parameters[ENDFOOT_TYPE].update({'modify': {'funct': scale_barcode,
                                               'kwargs': {}}})


class TNSSpaceColonizationGrower(TNSGrowerWrapper):

    def __init__(self, input_distributions, synthesis_parameters):
        raise NotImplementedError

    def add_synapse_data(self, point_cloud):
        self._parameters['context'] = {'point_cloud': point_cloud}
