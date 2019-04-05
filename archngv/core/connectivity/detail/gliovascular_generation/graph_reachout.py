""" Contains strategies that can be deployed by astrocytes in order to
determine the sites of contact with the vasculature
"""
import logging
from functools import partial

import numpy


L = logging.getLogger(__name__)


def maximum_reachout(val_arr, n_classes):
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
    min_cluster_idx[1:] = sorted_idx[idx[1: -1] + 1]

    return min_cluster_idx


def random_selection(val_arr, n_classes):
    """ Returns a random subsample on n_classes elements from
    the distance array
    """
    idx = numpy.arange(len(var_arr))
    return numpy.random.choice(idx, size=n_classes, replace=False)


REACHOUT_STRATEGIES = {'maximum_reachout': maximum_reachout,
                       'random_selection': random_selection}


def deploy(input_strategy, available_strategies):
    """ Deploys a strategy function while checking its existence from 
    the dict of available strategies.
    """
    try:

        selected_strategy = available_strategies[input_strategy]
        L.info('Strategy %s is selected', input_strategy)

    except KeyError:

        available_strategies = list(available_strategies.keys())

        warn_msg = 'Strategy {} is not available.'.format(input_strategy)
        info_msg = 'Available strategies: {}'.format(available_strategies)
        L.warning(warn_msg + '\n' + info_msg)
        raise KeyError(warn_msg + '\n' + info_msg)

    return selected_strategy


strategy = partial(deploy, available_strategies=REACHOUT_STRATEGIES)
