import os
import time


def _create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class NGVConfig(object):
    @classmethod
    def create_template(cls, parent_directory, experiment_name, config_name='ngv_config'):
        config = {}

        config['experiment_name'] = experiment_name
        config['parent_directory'] = parent_directory

        config['output_paths'] = {
            'morphology': 'morphology',
            'figures': 'figures'
        }

        config['metadata'] = {'creation_data': time.strftime("%S%M%H")}

        config['input_paths'] = {}
        config['parameters'] = {}

        return cls(config, config_name)

    @classmethod
    def from_file(cls, file_path):
        import json

        with open(file_path, 'r') as fp:
            config_dict = json.load(fp)

        return cls(config_dict, 'ngv_config')

    def __init__(self, config_dict, config_name):
        self._config = config_dict
        self._name = config_name

    def create_directories(self):
        _create_dir(self.parent_directory)
        _create_dir(self.experiment_directory)
        _create_dir(self.morphology_directory)
        _create_dir(self.endfeetome_directory)
        _create_dir(self.intermediate_directory)
        _create_dir(self.spatial_index_directory)
        _create_dir(self.neuronal_data_directory)

        try:
            _create_dir(self.figures_directory)
        except KeyError:
            pass

    def __str__(self):
        return self._config.__str__()

    def __repr__(self):
        return self._config.__repr__()

    def add_input_path(self, entry, path):
        pwd_path = self.parent_directory
        self._config['input_paths'][entry] = os.path.join(pwd_path, path)

    def add_output_path(self, entry, path):
        exp_path = self.experiment_directory
        self._config['output_paths'][entry] = os.path.join(exp_path, path)

    @property
    def parameters(self):
        return self._config['parameters']

    def input_paths(self, key):
        return self._config['input_paths'][key]

    def output_paths(self, key):
        op = self._config['output_paths']
        return os.path.join(self.experiment_directory, op[key])

    @property
    def metadata(self):
        return self._config['metadata']

    @property
    def name(self):
        return self._name

    @property
    def self_path(self):
        return os.path.join(self.experiment_directory, self.name)

    @property
    def parent_directory(self):
        return self._config['parent_directory']

    @parent_directory.setter
    def parent_directory(self, directory):
        self._config['parent_directory'] = directory

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def experiment_directory(self):
        ename = self.experiment_name
        ppath = self.parent_directory
        return os.path.join(ppath, ename)

    @property
    def circuit_path(self):
        return self.output_paths('circuit')

    @property
    def morphology_directory(self):
        return self.output_paths('morphology')

    @property
    def endfeetome_directory(self):
        return self.output_paths('endfeetome')

    @property
    def intermediate_directory(self):
        return self.output_paths('intermediate')

    @property
    def spatial_index_directory(self):
        return self.output_paths('spatial_index')

    @property
    def neuronal_data_directory(self):
        return self.output_paths('neuronal_data')

    @property
    def figures_directory(self):
        return self.output_paths('figures')

    def save(self):
        import json
        file_path = self.self_path + '.json'
        with open(file_path, 'w') as fp:
            json.dump(self._config, fp, indent=4)

    def generate_circuit_config(self):
        string = 'Run Default\n'
        string += '{\n'
        string += '    MorphologyPath {}\n'.format(self.output_paths('morphology'))
        string += '    CircuitPath {}\n'.format(self.experiment_directory)
        string += '}'
        with open(self.output_paths('circuit_config'), 'w') as f:
            f.write(string)
