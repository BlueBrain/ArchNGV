
import logging
from .common import H5ContextManager


L = logging.getLogger(__name__)


class SynapticData(H5ContextManager):

    @property
    def synapse_coordinates(self):
        return self._fd['/synapse_coordinates']

    @property
    def n_synapses(self):
        return len(self.synapse_coordinates)


class SynapticDataInfo(SynapticData):

    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('synaptic_data')
        super(SynapticDataInfo, self).__init__(filepath)
        self._config = ngv_config
