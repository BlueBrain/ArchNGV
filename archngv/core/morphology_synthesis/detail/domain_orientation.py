import numpy
import numpy as np
import logging
from sklearn.cluster import KMeans

from morphmath import rowwise_dot

L = logging.getLogger(__name__)


def interpolate_triangles(points, triangles):
    new_points = (points[triangles[:, 0]] + points[triangles[:, 1]] + points[triangles[:, 2]]) / 3.
    return numpy.vstack((points, new_points))

def _angle(v1, v2):
    return np.arccos(np.dot(v1, v2))

def orientation_function(soma_center, domain, n_trunks,
                         predetermined_orientations=None,
                         return_lengths=False,
                         face_interpolation=True):
    """ Given the domain shape and the number of apical directions, find
    from the trunk directions. It uses kmeans of the vectors from the center
    to the convex hull vertices and from the resulting classes takes the furthest
    one.
    Returns the normalized directions for the apical trunks.
    """
    L.info('Domain Orientation Started.')

    if face_interpolation:
        vectors = interpolate_triangles(domain.points, domain.triangles) - soma_center
    else:
        vectors = domain.points - soma_center

    L.info('pred: {}'.format(predetermined_orientations))


    if predetermined_orientations is not None:

        L.debug('Number of Predetermined Orientations: {}'.format(len(predetermined_orientations)))

        predetermined_orientations /= np.linalg.norm(predetermined_orientations, axis=1)[:, np.newaxis]

        existing = list(predetermined_orientations)

        n_trunks -= len(existing)

    else:

        L.info('There are no predetermined orientations.')

        existing = []

    L.debug('Number of trunks to calculate: {}'.format(n_trunks))

    # domain vector distances and orientations
    distances = np.linalg.norm(vectors, axis=1)
    orientations = vectors / distances[:, np.newaxis]

    # all the available vectors to choose from
    idx = set(range(len(vectors)))

    # remove overlaps with predetermined orientations
    for i, ori in enumerate(orientations):
        for ex in existing:
            if np.allclose(ex, ori):
                idx.remove(i)

    results = set()

    while len(results) < n_trunks:

        total = 0.
        best_index = None

        for index in idx:

            if len(existing) == 0:

                current_sum = distances[index]

                if current_sum > total:
                    best_index = index
                    total = current_sum

            else:

                current_sum =  min([_angle(orientations[index], e) for e in existing])

                if current_sum > total:
                    best_index = index
                    total = current_sum


        assert best_index is not None

        results.add(best_index)
        existing.append(orientations[best_index])
        idx.remove(best_index)

    res_idx = np.asarray(list(results), dtype=np.intp)

    L.debug('Number of idx calculated: {}'.format(len(res_idx)))

    if return_lengths:
        L.info('Domain Orientation Started.')
        return orientations[res_idx], distances[res_idx]

    else:
        L.info('Domain Orientation Started.')
        return orientations[res_idx]
