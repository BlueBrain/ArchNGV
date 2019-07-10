import os
import sys

import bluepy
from archngv import NGVConfig
from archngv.core.exporters.\
    export_neuronal_somata_geometries import extract_neuronal_somata_information


def extract_neuronal_geometry(ngv_config, run_parallel):

    circuit_config = os.path.join(ngv_config.input_paths('microcircuit_path'), 'CircuitConfig')

    neuronal_microcircuit = bluepy.Circuit(circuit_config)

    extract_neuronal_somata_information(ngv_config.neuronal_data_directory, neuronal_microcircuit)


if __name__ == '__main__':

    cfg = NGVConfig.from_file(sys.argv[1])
    extract_neuronal_geometry(ngv_config, None)

