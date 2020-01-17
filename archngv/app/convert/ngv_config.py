'''
Export input / output paths to NGVConfig JSON
'''

import click


@click.command(help=__doc__)
@click.argument('build_dir')
@click.option('--bioname', help='Path to bioname folder', required=True)
@click.option('-o', '--output', help='Path to output file (JSON)', required=True)
def cmd(build_dir, bioname, output):
    # pylint: disable=missing-docstring
    import json
    import os.path

    from archngv.app.utils import load_yaml

    def _load_config(name):
        return load_yaml(os.path.join(bioname, '%s.yaml' % name))

    while build_dir.endswith('/'):
        build_dir = build_dir[:-1]

    build_dir = os.path.abspath(build_dir)

    bioname = os.path.abspath(bioname)

    common_config = _load_config('MANIFEST')['common']
    cell_placement_config = _load_config('cell_placement')
    synthesis_config = _load_config('synthesis')
    synthesis_config['endfeet_area_reconstruction'] = _load_config('endfeet_area')

    config = {
        'experiment_name': os.path.basename(build_dir),
        'parent_directory': os.path.dirname(build_dir),
        'input_paths': {
            'voxelized_intensity': os.path.join(common_config['atlas'], '%s.nrrd' % cell_placement_config['density']),
            'voxelized_brain_regions': os.path.join(common_config['atlas'], 'brain_regions.nrrd'),
            'vasculature': common_config['vasculature'],
            'microcircuit_path': os.path.dirname(common_config['base_circuit']),
            'synaptic_data': common_config['base_circuit_connectome'],
            'vasculature_mesh': common_config['vasculature_mesh'],
            'tns_astrocyte_distributions': os.path.join(bioname, 'tns_distributions.json'),
            'tns_astrocyte_parameters': os.path.join(bioname, 'tns_parameters.json'),
        },
        'output_paths': {
            'morphology': 'morphologies',
            'cell_data': 'cell_data.h5',
            'synaptic_connectivity': 'synaptic_connectivity.h5',
            'gliovascular_connectivity': 'gliovascular_connectivity.h5',
            'gliovascular_data': 'gliovascular_data.h5',
            'neuroglial_connectivity': 'sonata/edges/neuroglial.h5',
            'neuroglial_data': 'neuroglial_data.h5',
            'endfeet_areas': 'endfeet_areas.h5',
            'microdomain_mesh': 'microdomains/tesselation.stl',
            'microdomains': 'microdomains/microdomains.h5',
            'overlapping_microdomain_mesh': 'microdomains/overlapping_tesselation.stl',
            'overlapping_microdomains': 'microdomains/overlapping_microdomains.h5',
        },
        'parameters': {
            'cell_placement': _load_config('cell_placement'),
            'microdomain_tesselation': _load_config('microdomains'),
            'gliovascular_connectivity': _load_config('gliovascular_connectivity'),
            'neuroglial_connectivity': {},
            'synthesis': synthesis_config,
        },
    }
    with open(output, 'w') as f:
        json.dump(config, f, indent=2)
