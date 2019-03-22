""" Contains strategies that can be deployed by astrocytes in order to
determine the sites of contact with the vasculature
"""
import logging
from functools import partial

import numpy

L = logging.getLogger(__name__)


def _maximum_reachout(val_arr, n_classes):
    """ Uses jenks natural break algorithm to find the closest distance
    to each class of distances
    """
    import jenkspy

    sorted_idx = numpy.argsort(val_arr)
    sorted_arr = val_arr[sorted_idx]

    breaks = jenkspy.jenks_breaks(sorted_arr, int(n_classes))

    idx = numpy.where(numpy.in1d(sorted_arr, breaks))[0]

    min_cluster_idx = numpy.empty(n_classes, dtype=numpy.intp)

    min_cluster_idx[0] = sorted_idx[idx[0]]
    min_cluster_idx[1:] = sorted_idx[idx[1:-1] + 1]

    return min_cluster_idx


def _random_selection(val_arr, n_classes):
    """ Returns a random subsample on n_classes elements from
    the distance array
    """
    idx = numpy.arange(len(var_arr))
    return numpy.random.choice(idx, size=n_classes, replace=False)


_REACHOUT_STRATEGIES = {'maximum_reachout': _maximum_reachout,
                        'random_selection': _random_selection}


def _deploy(input_strategy, available_strategies):
    """ Deploys a strategy function while checking its existence from 
    the dict of available strategies.
    """
    try:

        strategy = available_strategies[input_strategy]
        L.info('Strategye {} is selected'.format(input_strategy))

    except KeyError:

        available_strategies = list(available_strategies.keys())

        msg = ''
        warn_msg = 'Strategy {} is not available.'.format(input_strategies)
        info_msg = 'Available strategies: {}'.format(available_strategies)
        L.warning('Strategy {} is not available.'.format(input_strategies))
        L.info('Available strategies: {}'.format(available_strategies))
        raise KeyError(warn_msg + '\n' + info_msg)

    return strategy


#domain_strategy = partial(deploy, available_strategies=DOMAIN_STRATEGIES)
strategy = partial(_deploy, available_strategies=_REACHOUT_STRATEGIES)
