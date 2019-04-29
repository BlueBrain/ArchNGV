import os
import json
import logging
from copy import deepcopy
import numpy as np

from .detail.tns_wrapper import TNSGrowerWrapper
from .detail.data_extraction import obtain_endfeet_data
from .detail.data_extraction import obtain_synapse_data
from .detail.data_extraction import obtain_cell_properties
from .detail.annotation import export_endfoot_location

from tns import extract_input


L = logging.getLogger(__name__)


def synthesize_endfeet_for_astrocyte(args):
    """ Synthesize the endfeet for one astrocyte
    """
    astrocyte_index, ngv_config = args

    parameters_path = ngv_config.input_paths('tns_astrocyte_parameters')
    distributions_path = ngv_config.input_paths('tns_astrocyte_distributions')

    targets = obtain_endfeet_data(ngv_config, astrocyte_index)

    if targets is not None:

        cell_name, soma_pos, soma_rad, microdomain = \
            obtain_cell_properties(ngv_config, astrocyte_index)

        # extract synapses coordinates
        synapses = obtain_synapse_data(ngv_config, astrocyte_index)

        # initialize tns grower adapter
        endfeet_grower = TNSGrowerWrapper(parameters_path, distributions_path)

        # set soma position and radius
        endfeet_grower.set_soma_properties(soma_pos, soma_rad)

        # constrain the grower to its microdomain extent
        endfeet_grower.add_microdomain_boundary(microdomain)

        # lambda function is passed from config as string
        lambda_string = \
            ngv_config.parameters['synthesis']['attraction_field']

        # add the targets to grow to and the attraction field
        # function which depends on the distance to the target
        endfeet_grower.set_endfeet_targets(targets, eval(lambda_string))

        radius_of_influence, \
        removal_radius = ngv_config.parameters['synthesis']['point_cloud']

        # add synapse coordinates as a point cloud
        endfeet_grower.add_point_cloud(synapses, radius_of_influence, removal_radius)

        # we need to make sure that our extracted homology will suffice
        # in order to reach the target. If they are shorter, we scale them.
        endfeet_grower.enable_endfeet_barcode_scaling()

        endfeet_grower.grow()

        output_file = os.path.join(ngv_config.morphology_directory, '{}.h5'.format(cell_name))
        endfeet_grower.write(output_file)

        # reload cell in readonly to extract indices to targets
        export_endfoot_location(output_file, targets)


def synthesize_astrocyte_endfeet(ngv_config, astrocyte_ids, apply_func):
    """ Launch the endfeet synthesizer with all astrocyte_ids using the respective
    apply_func
    """
    def data_generator(ngv_config, ids):
        for astro_id in ids:
            yield astro_id, ngv_config

    apply_func(
                synthesize_endfeet_for_astrocyte,
                data_generator(ngv_config, astrocyte_ids)
              )

