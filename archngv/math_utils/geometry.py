""" Functions related to geometry
"""
import numpy as np
from archngv.math_utils.linear_algebra import skew_symmetric_matrix


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
