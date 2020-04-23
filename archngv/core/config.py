""" NGV Project configuration container """

import os
import json
import logging


L = logging.getLogger(__name__)


def _create_dir(directory):
    """ Create given directory """
    if not os.path.exists(directory):
        os.makedirs(directory)


DEFAULT_CONFIG_PATH = 'build/ngv_config.json'


class NGVConfig(object):
    """ Configuration data structure for ArchGNV
    """
    @classmethod
    def _resolve_config(cls, arg):
        if isinstance(arg, str):
            if os.path.isfile(arg):
                config_path = arg
            elif os.path.isdir(arg):
                config_path = os.path.join(arg, DEFAULT_CONFIG_PATH)
            else:
                msg = 'Unknown argument type {}'.format(type(arg))
                L.error(msg)
                raise TypeError(msg)

            L.info('NGV config loaded from path %s', config_path)
            return cls.from_file(config_path)

        try:
            assert isinstance(arg, NGVConfig)
        except AssertionError:
            msg = 'Incompatible argument of type {}'.format(type(arg))
            L.error(msg)
            raise TypeError(msg)

        return arg

    @classmethod
    def from_file(cls, file_path):
        """ Load a json config from file """
        with open(file_path, 'r') as fp:
            config_dict = json.load(fp)
        return cls(config_dict, 'ngv_config')

    def __init__(self, config_dict, config_name):
        self._config = config_dict
        self._name = config_name

    def create_directories(self):
        """ Create directory hierarchy """
        _create_dir(self.parent_directory)
        _create_dir(self.experiment_directory)
        _create_dir(self.morphology_directory)
        _create_dir(self.intermediate_directory)
        _create_dir(self.spatial_index_directory)
        _create_dir(self.neuronal_data_directory)

        try:
            _create_dir(self.figures_directory)
        except KeyError:
            pass

    def __str__(self):
        """ str representation """
        return json.dumps(self._config, indent=4)

    def __repr__(self):
        """ str repr """
        return self._config.__repr__()

    def add_input_path(self, entry, path):
        """ Add input path to the config
        Args:
            entry: string
                key name of the input path
            path: string
                absolute path to the input data
        """
        pwd_path = self.parent_directory
        self._config['input_paths'][entry] = os.path.join(pwd_path, path)

    def add_output_path(self, entry, path):
        """
        Args:
            entry: string
                key name of the output path
            path: string
                relative path to the project's root
        """
        exp_path = self.experiment_directory
        self._config['output_paths'][entry] = os.path.join(exp_path, path)

    @property
    def parameters(self):
        """ Configuration parameters """
        return self._config['parameters']

    def input_paths(self, key):
        """ Get input path from key """
        return self._config['input_paths'][key]

    def output_paths(self, key):
        """ Get output path from key """
        op = self._config['output_paths']
        return os.path.join(self.experiment_directory, op[key])

    @property
    def metadata(self):
        """ Config Metadata """
        return self._config['metadata']

    @property
    def name(self):
        """ Config Name """
        return self._name

    @property
    def self_path(self):
        """ Config path """
        return os.path.join(self.experiment_directory, self.name)

    @property
    def parent_directory(self):
        """ Project root directory """
        return self._config['parent_directory']

    @parent_directory.setter
    def parent_directory(self, directory):
        """ Parent directory setter """
        self._config['parent_directory'] = directory

    @property
    def experiment_name(self):
        """ Experiment Name """
        return self._config['experiment_name']

    @property
    def experiment_directory(self):
        """ Experiment's directory """
        ename = self.experiment_name
        ppath = self.parent_directory
        return os.path.join(ppath, ename)

    @property
    def circuit_path(self):
        """ Path to the circuit """
        return self.output_paths('circuit')

    @property
    def morphology_directory(self):
        """ Path to the morphology directory """
        return self.output_paths('morphology')

    @property
    def intermediate_directory(self):
        """ Path to the intermediate directory """
        return self.output_paths('intermediate')

    @property
    def spatial_index_directory(self):
        """ Directory where the spatial indexes are stored """
        return self.output_paths('spatial_index')

    @property
    def neuronal_data_directory(self):
        """ Neuronal data directory """
        return self.output_paths('neuronal_data')

    @property
    def figures_directory(self):
        """ Figures directory """
        return self.output_paths('figures')

    def save(self):
        """ Write config to file """
        file_path = self.self_path + '.json'
        with open(file_path, 'w') as fp:
            json.dump(self._config, fp, indent=4)

    def generate_circuit_config(self):
        """ Create a CircuitConfig file """
        string = 'Run Default\n'
        string += '{\n'
        string += '    MorphologyPath {}\n'.format(self.output_paths('morphology'))
        string += '    CircuitPath {}\n'.format(self.experiment_directory)
        string += '}'
        with open(self.output_paths('circuit_config'), 'w') as f:
            f.write(string)
