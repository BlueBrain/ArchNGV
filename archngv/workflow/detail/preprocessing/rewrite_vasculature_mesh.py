
import h5py
import logging
import numpy as np
from archngv import NGVConfig

L = logging.getLogger(__name__)


def rewrite_vasculature(ngv_config, map_func):

    import openmesh
    from openmesh import write_mesh


    filepath = config.input_paths('vasculature_mesh')

    L.info('Loading Mesh')

    mesh = openmesh.read_trimesh(filepath)

    L.info('Writing Mesh')

    ext = os.path.splittext(filepath)

    new_filepath = filepath.replace(ext, '_fixed' + ext)

    write_mesh(new_filepath, mesh) 


if __name__ == '__main__':

    import sys

    config = NGVConfig.from_file(sys.argv[1])

    rewrite_vasculature(config, None)