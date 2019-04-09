import os
import logging
from archngv.core.exporters import export_neuron_synapse_association

L = logging.getLogger(__name__)

def create_neuron_synapse_maps(ngv_config, run_parallel):

    import bluepy

    path = os.path.join(ngv_config.input_paths('microcircuit_path'), 'CircuitConfig')
    circuit = bluepy.Circuit(path)

    L.info('Neuronal Circuit: {}'.format(path))

    L.info('Started exporting associations.')
    export_neuron_synapse_association(
        circuit.v2,
        output_path_connectivity=ngv_config.output_paths('synaptic_connectivity'),
        output_path_data=ngv_config.output_paths('synaptic_data')
    )
