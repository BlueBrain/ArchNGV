""" Contains strategies that can be deployed by astrocytes in order to
determine the sites of contact with the vasculature
"""
import logging
import numpy as np

from archngv.exceptions import NGVError
from archngv.utils.linear_algebra import rowwise_dot


L = logging.getLogger(__name__)


def _argsort_components(source, components):
    """
    Sorts the N components with respect to the distance of the closest point
    in each component to the source.

    Args:
        source: array[float, (3,)]
        components list[pandas.DataFrame]:
            A list of dataframes containing the positions of the connected component
            elements.

    Returns:
        sorted_indices ndarray: int[(N,)]
            The sorted indices that correspond to the components array
        closest_vertices ndarray: int[(N,)]
            The closest global vertex of each component.
    """
    closest_vertices = np.empty(len(components), dtype=np.int)
    closest_distances = np.empty(len(components), dtype=np.float)

    for i, comp in enumerate(components):

        points = comp.loc[:, ('x', 'y', 'z')]

        distances = np.linalg.norm(source - points, axis=1)
        closest_pos = np.argmin(distances)

        closest_vertices[i] = comp.index[closest_pos]
        closest_distances[i] = distances[closest_pos]

    return np.argsort(closest_distances), closest_vertices


def _distribute_elements_in_buckets(n_elements, bucket_capacities):
    """
    Distributes n_elements into k buckets with specific capacities.

    Step 1: Split the n_elements into K buckets via the whole part of the
    division n_elements // n_buckets. We keep the remainder.

    Step 2: Some buckets have smaller capacity, so the elements we distributed
    in step 1 will overflow. Find these buckets and remove the extra elements from
    them and add them to the remainder.

    Step 3: Starting from the left to the right iterate over the buckets and redistribute
    the remainder to the rest of the buckets that are not full, filling each bucket in
    sequence.

    Args:
        n_elements int:
            Number of elements to be distributed in K buckets
        bucket_capacities ndarray: int[(K,)]

    Returns:
        elements_per_bucket ndarray: int[(K,)]
            Number of elements per bucket, respecting each capacity. It should
            hold that sum(elements_per_bucket) = n_elements

    Notes:
        The number of elements should not exceed the total bucket capacity, i.e.:
            sum(bucket_capacities) >= n_elements

        Assumption:
            The buckets should be sorted from the most to the least significant.
    """
    n_buckets = len(bucket_capacities)

    quotient, remainder = divmod(n_elements, n_buckets)

    elements_per_bucket = np.full(n_buckets, fill_value=quotient, dtype=np.int)

    # find buckets, the capacity of which is smaller than the number
    # of elements they currently have
    are_overflowing_mask = elements_per_bucket > bucket_capacities

    # count the extra elements
    remainder += np.sum(elements_per_bucket[are_overflowing_mask] - bucket_capacities[are_overflowing_mask])

    # set the size of the buchets to their maximum capacity
    elements_per_bucket[are_overflowing_mask] = bucket_capacities[are_overflowing_mask]

    # iterate over the buckets and fill the non-full ones with
    # the rest of the elements that couldn't fit in the smaller ones
    for i, capacity in enumerate(bucket_capacities):
        if remainder == 0:
            break

        remaining_space = capacity - elements_per_bucket[i]
        if remaining_space >= remainder:
            elements_per_bucket[i] += remainder
            remainder = 0
        else:
            elements_per_bucket[i] = capacity
            remainder -= remaining_space

    return elements_per_bucket


def _select_component_targets(source, comp, n_elements):
    """
    Selects n_elements from the component comp points.

    Args:
        source ndarray: float[(3,)]
            The coordinates of the source node which represents the position
            of the astrocytic soma.
        comp pandas.DataFrame:
            A dataframe that contains the coordinates of the potential targets
            we want to choose from.

        n_elements: int
            Number of elements to choose.

    Returns:
        target_indices: array[int, (n_elements,)]

    Notes:
        First point is always the closest. Every next point maximizes
        the minimun distance to the previously selected points in the loop.
    """
    points = comp.loc[:, ('x', 'y', 'z')].to_numpy(copy=False)

    first_pos = np.argmin(np.linalg.norm(source - points, axis=1))

    occupied = {first_pos}
    available = set(range(len(points)))
    available.remove(first_pos)

    vectors = points - points[first_pos]
    scores = rowwise_dot(vectors, vectors)

    for _ in range(n_elements - 1):

        # find the available index that has max respective score
        best_index = max(available, key=lambda k: scores[k])

        occupied.add(best_index)
        available.remove(best_index)

        for f_index in available:
            vec = points[f_index] - points[best_index]
            # update the distance to the closest point
            # if the new point is not the closest the previous one is kept
            scores[f_index] = min(np.dot(vec, vec), scores[f_index])

    occupied = np.asarray(list(occupied))
    return comp.index[occupied]


def _maximum_reachout(source, targets, n_classes):
    """
    Args:
        source ndarray: float[(3,)]
            Coordinates represeting the astrocyte's soma position.
        targets pandas.DataFrame:
            A dataframe with the coordinates of the potential targets we want to choose from.
        n_classes int:
            Number of classes to return

    Returns:
        selected_indices: array[int, (n_classes,)]

    Notes:
    Given an array of available targets select a subset of the of size n_classes, according
    to the following algorithm:

    1. Create groups (sections, components) of the points based on their assigned section id.

    2. Sort the groups with respect to the closest point inside each component to the source.

    3. If the classes are fewer than the components, return the closest ones.

    4. Otherwise, distribute the number of classes to each component taking into account how
       many points it has available. Components on the left have high priority of being assigned
       more points if their capacity allows it. If all components are big enough then they share
       an equal amount classes.

    5. The first point within each component is always the closest one. Every other point is selected
       so as to maximize its distance with all other selected points so far in the iteration.
    """
    components = [df for _, df in targets.groupby('vasculature_section_id')]
    sorted_indices, closest_vertices = _argsort_components(source, components)

    # assign the closest components if we have less classes than comps
    if n_classes <= len(components):
        return closest_vertices[sorted_indices[:n_classes]]

    # sort components by closest point inside each compoenent to the source
    sorted_components = [components[sorted_index] for sorted_index in sorted_indices]
    component_sizes = np.fromiter(map(len, sorted_components), np.int)

    # find how many targets we will have in each component given the number their available
    # points
    n_elements_per_component = _distribute_elements_in_buckets(n_classes, component_sizes)

    selected = np.empty(n_classes, dtype=np.int)

    n = 0
    for comp, n_elements in zip(sorted_components, n_elements_per_component):

        selected[n: n + n_elements] = _select_component_targets(source, comp, n_elements)
        n += n_elements

    return selected


def _random_selection(_, targets, n_classes):
    """ Returns a random subsample on n_classes elements from
    the distance array
    """
    n_classes = min(n_classes, len(targets))
    idx = np.random.choice(np.arange(len(targets)), size=n_classes, replace=False)
    return targets.index[idx]


REACHOUT_STRATEGIES = {'maximum_reachout': _maximum_reachout,
                       'random_selection': _random_selection}


def strategy(input_strategy):
    """ Deploys a strategy function while checking its existence from
    the dict of available strategies.
    """
    if input_strategy not in REACHOUT_STRATEGIES:
        msg = (
            'Strategy {} is not available.'.format(input_strategy) + '\n' +
            'Available strategies: {}'.format(REACHOUT_STRATEGIES)
        )
        raise NGVError(msg)

    selected_strategy = REACHOUT_STRATEGIES[input_strategy]
    L.info('Strategy %s is selected', input_strategy)
    return selected_strategy
