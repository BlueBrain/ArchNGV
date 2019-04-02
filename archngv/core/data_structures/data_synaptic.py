""" Synaptic Data Structures
"""
from .common import H5ContextManager


class SynapticData(H5ContextManager):
    """ Synaptic data structure """
    @property
    def synapse_coordinates(self):
        """ Coordinates xyz """
        return self._fd['/synapse_coordinates']

    @property
    def n_synapses(self):
        """ Number of synapses """
        return len(self.synapse_coordinates)


class SynapticDataInfo(SynapticData):
    """ Rich synaptic data structure """
    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('synaptic_data')
        super(SynapticDataInfo, self).__init__(filepath)
        self._config = ngv_config
