""" Neuroglial Connectivity
"""

import logging

from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_synaptic import SynapticData
from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
from archngv.core.data_structures.connectivity_synaptic import SynapticConnectivity
from archngv.core.connectivity.neuroglial_generation import generate_neuroglial
from archngv.core.exporters.export_neuroglial_connectivity import export_neuroglial_connectivity


L = logging.getLogger(__name__)


#@run_spatial_index_on_SSDs
def create_neuroglial_connectivity(ngv_config, run_parallel):

    with \
        SynapticData(ngv_config.output_paths('synaptic_data')) as synaptic_data, \
        SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity')) as synaptic_connectivity, \
        CellData(ngv_config.output_paths('cell_data')) as cell_data, \
        MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as microdomains:

        astrocyte_ids = cell_data.astrocyte_gids[:]

        L.info('Generating neuroglial connectivity')
        data_iterator = generate_neuroglial(astrocyte_ids,
                                            microdomains,
                                            synaptic_data,
                                            synaptic_connectivity)

        n_unique_synapses = synaptic_connectivity.n_synapses
        n_unique_neurons = synaptic_connectivity.n_neurons
        n_unique_astrocytes = len(astrocyte_ids)

        neuroglial_connectivity_path = \
            ngv_config.output_paths('neuroglial_connectivity')

        L.info('Exporting the per astrocyte files...')
        export_neuroglial_connectivity(data_iterator,
                                       n_unique_astrocytes,
                                       n_unique_synapses,
                                       n_unique_neurons,
                                       neuroglial_connectivity_path)


"""
def ensure_index_existence(ngv_config):

    # path to the index director on the HDD
    si_synapses_path_hdd = ngv_config.output_paths('synapses_index')
    L.info('Index HDD location: {}'.format(si_synapses_path_hdd))

    filepath_idx_hdd = si_synapses_path_hdd + '.idx'
    filepath_dat_hdd = si_synapses_path_hdd + '.dat'

    assert os.path.isfile(filepath_idx_hdd), "{} does not exist.".format(filepath_idx_hdd)
    assert os.path.isfile(filepath_dat_hdd), "{} does not exist.".format(filepath_dat_hdd)


def run_spatial_index_on_SSDs(old_function):

    def new_function(*args, **kwargs):

        ngv_config = args[0]

        # index file extension
        idx_ext = '.idx'

        # data file extension
        dat_ext = '.dat'

        # path to the index director on the HDD
        si_synapses_path_hdd = ngv_config.output_paths('synapses_index')
        L.info('Index HDD location: {}'.format(si_synapses_path_hdd))

        filepath_idx_hdd = si_synapses_path_hdd + idx_ext
        filepath_dat_hdd = si_synapses_path_hdd + dat_ext

        assert os.path.isfile(filepath_idx_hdd), "{} does not exist.".format(filepath_idx_hdd)
        assert os.path.isfile(filepath_dat_hdd), "{} does not exist.".format(filepath_dat_hdd)

        assert "SLURM_JOB_ID" in os.environ, "There is no allocation available"
        assert os.path.isdir('/nvme'), '/nvme is not accessible. Have you allocated with -Cnvme?'

        # destination of the copied index on the SSDs
        si_synapses_path_ssd = '/nvme/{}/{}/si_synapses'.format(getuser(),
                                                                os.environ["SLURM_JOB_ID"])
        L.info('Index SSD destination: {}'.format(si_synapses_path_ssd))

        # filepaths in the nvme ssd directory
        filepath_idx_ssd = si_synapses_path_ssd + idx_ext
        filepath_dat_ssd = si_synapses_path_sss + dat_ext

        # copy the index for the hdd to the ssd
        os.system("rsync --ignore-existing {} {}".format(filepath_idx_hdd, filepath_idx_ssd))
        os.system("rsync --ignore-existing {} {}".format(filepath_dat_hdd, filepath_dat_ssd))

        # symbolic link filenames
        symbolic_link_ssd_to_hdd_dat = si_synapses_path_hdd + '_ssd.dat'
        symbolic_link_ssd_to_hdd_idx = si_synapses_path_hdd + '_ssd.idx'

        try:
            # unlink any existing stray links
            os.unlink(symbolic_link_ssd_to_hdd_dat)
            os.unlink(symbolic_link_ssd_to_hdd_idx)

        except OSError:

            L.info('No existing links found.')

        # create symbolic link in order for the files to be seen local to the project
        os.symlink(filepath_dat_ssd, symbolic_link_ssd_to_hdd_dat)
        os.symlink(filepath_idx_ssd, symbolic_link_ssd_to_hdd_idx)

        # add the linked ssd index in the config. The project is not aware of the switching
        ngv_config._config['output_paths']['synapses_index'] += '_ssd'

        L.info('SSD synapses index path: {}'.format(ngv_config.output_paths('synapses_index')))

        # run the main function
        old_function(*args, **kwargs)

        # remove the temp link and add back the hdd location of the index
        ngv_config._config['synapses_index'] = si_synapses_path_hdd

        # remove symbolic links
        os.unlink(symbolic_link_ssd_to_hdd_dat)
        os.unlink(symbolic_link_ssd_to_hdd_idx)

    return new_function
"""
