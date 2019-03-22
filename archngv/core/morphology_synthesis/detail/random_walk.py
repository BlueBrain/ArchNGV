import logging
import numpy as np

from morphmath import uniform_cartesian_unit_vectors


L = logging.getLogger(__name__)


def biased_random_walk(start, target, randomness, segment_length):
    """
    Grow a directed random walk from start to target
    with a specified randomness [0,1] (line, pure rw)
    """
    NMAX = 1000
    points = np.empty((NMAX, 4))

    # set as first point the starting point
    points[0] = start

    n = 1
    r = segment_length

    # stop when the target is reached
    while np.linalg.norm(target - points[n - 1, (0, 1, 2)]) > segment_length and n < NMAX - 1:

        # add the random orientation unit vector with 
        # the unite vector resulting from the tropism application
        u_s = uniform_cartesian_unit_vectors(k=1)[0]

        p = target - points[n - 1, (0, 1, 2)]

        t = randomness * u_s + (1. - randomness) * p / np.linalg.norm(p)

        n_t = np.linalg.norm(t)

        # create the point with the random direction
        points[n, (0, 1, 2)] = points[n - 1, (0, 1, 2)] + r * t / n_t
        points[n, 3] = points[0, 3]

        n += 1

    if n < NMAX - 1:

        points[n, (0, 1, 2)] = target
        points[n, 3] = points[0, 3]

        n += 1

    else:

        L.warning('Maximum number of iterations reached before hitting target.')

    return points[:n]

