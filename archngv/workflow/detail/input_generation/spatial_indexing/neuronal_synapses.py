import os
import logging
import resource
from getpass import getuser

from archngv.data_structures.ngv_data import SynapticData


L = logging.getLogger(__name__)


def create_synapses_spatial_index(ngv_config, map_func):

    filepath = ngv_config.output_paths('synapses_index')

    ssd_path = '/nvme/{}/{}'.format(getuser(), os.environ["SLURM_JOB_ID"])
    intermediate_ssd_target = '{}/si_synapses'.format(ssd_path)

    os.chdir(ssd_path)

    L.info('Current directory: {}'.format(os.getcwd()))

    try:
        os.remove(filepath + '.dat')
        os.remove(filepath + '.idx')
    except OSError:
        # already removed, all good
        pass

    try:
        os.remove(intermediate_ssd_target + '.dat')
        os.remove(intermediate_ssd_target + '.idx')
    except OSError:
        # already removed, all good
        pass

    with SynapticData(ngv_config.output_paths('synaptic_data')) as syn_data:

        L.info('Number of Synapses: {}'.format(len(syn_data.synapse_coordinates)))

        L.info('Target path: {}'.format(filepath))
        L.info('Creating the spatial index on SSDs: {}'.format(intermediate_ssd_target))

        # create the index on the ssds
        si = spatial_index(intermediate_ssd_target)
        index = si.create_from_points(syn_data.synapse_coordinates)

        # the index has to exit otherwise it keeps the files open and they look like empty!
        del index


    os.system('rsync -av {} {}'.format(intermediate_ssd_target + '.dat', filepath + '.dat'))
    L.info('Copied from si_synapses.dat {} to {}'.format(intermediate_ssd_target + '.dat', filepath + '.dat'))

    os.system('rsync -av {} {}'.format(intermediate_ssd_target + '.idx', filepath + '.idx'))
    L.info('Copied from si_synapses.idx {} to {}'.format(intermediate_ssd_target + '.idx', filepath + '.idx'))


