""" High level Circuit container. All NGV data can be accesed from here. """

import logging

from archngv.core.data_ngv import NGVData
from archngv.core.connectivity_ngv import NGVConnectome


L = logging.getLogger(__name__)


class NGVCircuit(object):
    """ Encapsulates all the information concerning an NGV
    circuit.

    Attributes:
        config: NGVConfig
        data: NGVData
        connectome: NGVConnectome
    """
    def __init__(self, ngv_config):
        self.config = ngv_config
        self.data = NGVData(ngv_config)
        self.connectome = NGVConnectome(ngv_config)

    @property
    def neuronal_microcircuit_path(self):
        """ Path to bbp neuronal microcircuit """
        return self.config.input_paths('neuronal_microcircuit')
