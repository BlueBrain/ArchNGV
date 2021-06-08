""" Functions related to geometry
"""
import numpy as np
from archngv.utils.linear_algebra import skew_symmetric_matrix


def apply_rotation_to_points(points, rotation_matrix):
    """
    Args:
        points: 2D array
            Row stacked 3D points
        rotation_matrix: 2D array
            3x3 Rotation Matrix

    Reutnrs:
        2D array:
            Rotated points using the rotation_matrix. The
            points are not centered to the origin.
    """
    return np.einsum('ij,kj->ik', points, rotation_matrix)


def uniform_spherical_angles(number_of_angles=1):
    """
    Args:
        number_of_angles: int
            Number of angles to generate
    Returns:
        1D array, 1D array:
            Arrays of size number_of_angles storing the angles phi and theta.
    """
    phi = np.random.uniform(0., 2. * np.pi, number_of_angles)
    theta = np.arccos(np.random.uniform(-1., 1., number_of_angles))
    return phi, theta


def uniform_cartesian_unit_vectors(number_of_vectors=1):
    """
    Args:
        number_of_vectors: int
            Number of angles to generate
    Returns:
        1D array, 1D array, 1D array:
            The x, y, z of the unit vectors corresponding to number_of_vectors
            uniformly on the surface of the unit sphere.
    """
    phi, theta = uniform_spherical_angles(number_of_vectors)

    sn_theta = np.sin(theta)

    return np.cos(phi) * sn_theta, np.cos(theta), np.sin(phi) * sn_theta


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, zeta = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    zeta = zeta * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    radius = np.sqrt(zeta)
    v_vec = (np.sin(phi) * radius, np.cos(phi) * radius, np.sqrt(2.0 - zeta))

    sine_value = np.sin(theta)
    cosine_value = np.cos(theta)

    r_m = np.array(((cosine_value, sine_value, 0),
                    (-sine_value, cosine_value, 0),
                    (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    return np.dot(np.outer(v_vec, v_vec) - np.eye(3), r_m)


def rotate_from_unit_vector_to_another(u_a, u_b):
    """
    Args:
        u_a: 1D array
        u_b: 1D array

    Returns:
        2D array:
            3x3 Rotation matrix from one vector to another.
    """
    v_x = skew_symmetric_matrix(np.cross(u_a, u_b))

    return np.identity(3) + v_x + np.linalg.matrix_power(v_x, 2) * (1. / (1. + np.dot(u_a, u_b)))


def rodrigues_rotation_matrix(axis, angle):
    '''
    Generates transformation matrix from unit vector
    and rotation angle. The rotation is applied in the direction
    of the axis which is a unit vector following the right hand rule.
    Inputs :
        axis : unit vector of the direction of the rotation
        angle : angle of rotation in rads
    Returns : 3x3 Rotation matrix
    '''
    def _sin(value):
        '''sine with case for pi multiples'''
        return 0. if np.isclose(np.mod(value, np.pi), 0.) else np.sin(value)

    sin_val = _sin(angle)
    cos_val = np.cos(angle)

    ss_m = skew_symmetric_matrix(axis / np.linalg.norm(axis))

    return np.identity(3) + sin_val * ss_m + (1. - cos_val) * np.linalg.matrix_power(ss_m, 2)


def sort_points(points):
    """
    Returns a lexicographic sorting of the 3D coordinates first by x
    then by y and finally by z.
    Args:
        points: array[float, (N, 3)]

    Returns:
        sorted_indices: array[int, (N,)]
    """
    return np.lexsort((points[:, 2], points[:, 1], points[:, 0]))


def are_consecutive_pairs_different(points):
    """
    Given an array of points in lexicographic order it returns
    a mask where the i-th entry is True if it is identical with
    the entry i-1. By construction the first entry is always different.

    Example:

        points = [[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1],
                  [0.1, 0.2, 0.3],
                  [0.1, 0.2, 0.3]]

        result: [True, False, True, False]
    """
    are_diff = np.empty(len(points), dtype=bool)
    are_diff[0] = True

    # check if the difference of the i-th and (i-1)th entries is zero
    are_close = np.isclose(points[1:] - points[:-1], 0.0)

    # if any coordinate xyz is not close, then the points are different
    are_diff[1:] = (~are_close).any(axis=1)
    return are_diff


def unique_points(points, decimals):
    """

    Args:
        points: array[float, (N, 3)]
            The array of points.
        decimals: int
            Points are rounded before sorting. This determines the number
            of significant digits.

    Returns:
        unique_indices: array[int, (M,)]
            The indices of the unique indices in the points array in the order that
            they appear in the array.
        inverse_mapping: array[int, (N,)]
            The mapping from the unique points back to the points array in the order
            they appear in the points array.

    Notes:

        inspired by numpy's unique1d
        https://docs.scipy.org/doc/numpy-1.3.x/reference/generated/numpy.unique1d.html

        Both unique and inverse mapping indices maintain the order of each point in the
        points array. Therefore this function is not the same as numpy's unique which does
        not maintain the order.
    """
    rounded_points = points.round(decimals=decimals)
    sorted_idx = sort_points(rounded_points)

    # check if each ordered points row is close to the one above
    is_unique = are_consecutive_pairs_different(rounded_points[sorted_idx])

    # consecutive duplicates has the same unique id
    duplicate_ids = np.cumsum(is_unique) - 1

    # maps the i-th row of points to the j-th row of sorted points
    to_sorted = np.argsort(sorted_idx)

    # maps each point in the initial array to the unique one
    inverse_mapping = np.empty_like(sorted_idx)

    # first point that appears in point array is the first indexed
    inverse_mapping[0] = 0

    # to keep track of the ids we visited so far
    # because the duplicat_ids is calculated on the sorted array
    # we need the to_sorted in order to go to the sorted indice
    visited = {duplicate_ids[to_sorted[0]]: 0}

    n = 1
    for i in range(1, len(sorted_idx)):

        s_index = to_sorted[i]
        cid = duplicate_ids[s_index]

        # if our point is unique or if we encounter
        # the first of the duplicate points, a unique
        # index n is assigned
        if is_unique[s_index] or cid not in visited:

            inverse_mapping[i] = n

            # keep track of the duplicate
            visited[cid] = n
            n += 1

        # the points is not unique and the first of the
        # duplicates is already registered
        else:

            # use the id of the first duplicate
            inverse_mapping[i] = visited[cid]

    # unique ids are sorted to maintain the order they appear
    # in the points array
    return np.sort(sorted_idx[is_unique]), inverse_mapping
