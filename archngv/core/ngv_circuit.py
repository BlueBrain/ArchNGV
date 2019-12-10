""" High level Circuit container. All NGV data can be accesed from here. """

from cached_property import cached_property

from archngv import NGVConfig
from archngv.core.data_ngv import NGVData
from archngv.core.connectivity_ngv import NGVConnectome


class NGVCircuit:
    """ Encapsulates all the information concerning an NGV
    circuit.

    Attributes:
        config: NGVConfig
    """
    def __init__(self, ngv_config):
        self._config = NGVConfig._resolve_config(ngv_config)

    def __repr__(self):
        """ Representation of circuit """
        return '<NGVCircuit @ {}>'.format(self.config.parent_directory)

    @property
    def config(self):
        """ NGV Config """
        return self._config

    @cached_property
    def data(self):
        """ Circuit data object """
        return NGVData(self.config)

    @cached_property
    def connectome(self):
        """ Circuit connectome object """
        return NGVConnectome(self.config)

    @property
    def neuronal_microcircuit_path(self):
        """ Path to bbp neuronal microcircuit """
        return self.config.input_paths('neuronal_microcircuit')
