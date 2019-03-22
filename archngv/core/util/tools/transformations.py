import numpy as np
from ..math import rodrigues_rotation_matrix

def scale(points, sx, sy, sz):

    return np.dot(points, np.diag((sx, sy, sz)))


def translate(points, tx, ty, tz):

    return points + (tx, ty, tz)


def affine(points, M):

    return np.einsum('ij,kj->ik', points, M)


def swap_axes(points, saxis1, saxis2):

    axes = {'x': 0, 'y': 1, 'z': 2}

    axis1 = axes[saxis1]
    axis2 = axes[saxis2]

    new_points = points.copy()

    new_points[:, axis1] = points[:, axis2]
    new_points[:, axis2] = points[:, axis1]

    return new_points

def center_xz(points):
    from scipy.spatial import ConvexHull

    new_points = points.copy()

    ch = ConvexHull(points[:, (0, 2)])

    new_points[:, (0, 2)] -= points[ch.vertices, :][:, (0, 2)].mean()

    return new_points

def rotate(points, axis, angle, origin=None):
    '''
    Rotation around unit vector following the right hand rule
    Parameters:
        obj : obj to be rotated (e.g. neurite, neuron).
            Must implement a transform method.
        axis : unit vector for the axis of rotation
        angle : rotation angle in rads
    Returns:
        A copy of the object with the applied translation.
    '''
    R = rodrigues_rotation_matrix(axis, angle)

    return np.einsum('ij,kj->ki', R, points)