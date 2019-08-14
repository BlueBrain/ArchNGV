'''area fitting'''
import logging

from archngv.core.endfeet_area_reconstruction.detail import _area_fitting


L = logging.getLogger(__name__)


def fit_area(endfoot, target_area):
    '''take and endfoot, and try to reduce it to target_area'''
    try:
        assert 0. < target_area < endfoot.area
        L.info('Started surface area reduction: Target: %s, Current: %s',
               target_area, endfoot.area)

        to_remove = _area_fitting.reduce_surface_area(endfoot,
                                                      target_area,
                                                      endfoot.extra['vertex']['travel_times'])

        L.debug('N Vertices to be removed: %d', len(to_remove))
        endfoot.shrink(to_remove)
        L.debug('AFTER: Actual Area: %s, Target Area: %s',
                endfoot.area, target_area)
    except AssertionError:
        L.info('Aborted surface area reduction')

    L.info('Endfoot Area Fitting completed. Target: %s, Current: %s',
           target_area, endfoot.area)
    return endfoot
