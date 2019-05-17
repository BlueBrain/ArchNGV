import logging

from .data_ngv import NGVData
from .connectivity_ngv import NGVConnectome


L = logging.getLogger(__name__)


class NGVCircuit(object):
    def __init__(self, ngv_config):
        self.config = ngv_config
        self.data = NGVData(ngv_config)
        self.connectome = NGVConnectome(ngv_config)

    @property
    def neuronal_microcircuit_path(self):
        return self.config.input_paths('neuronal_microcircuit')
