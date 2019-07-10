import logging

from archngv.core.data_structures.vasculature_morphology import Vasculature

from .spatial_index_adapter import spatial_index

L = logging.getLogger(__name__)

def vasculature_spatial_index(ngv_config, map_func):

    L.info('Loading vasculature dataset.')

    vasculature = Vasculature.load(ngv_config.input_paths('vasculature'))

    L.info("Vasculature Spatial Index is being generated.")

    vsi_filename = ngv_config.output_paths('vasculature_index')

    si = spatial_index(vsi_filename)

    return si.create_from_spheres(vasculature.points, vasculature.radii)


if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config_filepath = sys.argv[1]

    config = NGVConfig.from_file(config_filepath)

    vasculature_spatial_index(config, map)
