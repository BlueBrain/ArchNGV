import os
import logging
from archngv.core.exporters import export_neuron_synapse_association

L = logging.getLogger(__name__)

def create_neuron_synapse_maps(ngv_config, run_parallel):

    import bluepy

    path = os.path.join(ngv_config.input_paths('microcircuit_path'), 'CircuitConfig')
    circuit = bluepy.Circuit(path)

    L.info('Neuronal Circuit: {}'.format(path))

    connectome = circuit.v2.connectome
    neuronal_gidx = circuit.v2.cells.ids()

    L.info('Started exporting associations.')

    export_neuron_synapse_association(
                                        ngv_config,
                                        neuronal_gidx,
                                        connectome
                                     )

