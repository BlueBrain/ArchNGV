""" Functions related to geometry
"""
import math
import numpy as np
from .linear_algebra import normalize_vectors
from .linear_algebra import skew_symmetric_matrix
from .linear_algebra import rowwise_dot


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


def vectorized_triangle_normal(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a triangle.
        vectors2: 2D array
            Vectors where each row represents second side a triangle.

    Returns:
        2D array:
            Normal vectors of the triangle faces.
    """
    crosses = np.cross(vectors1, vectors2)
    return normalize_vectors(crosses)


def vectorized_parallelogram_area(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelogram.
        vectors2: 2D array
            Vectors where each row represents second side a paralellogram.
        The vectors must not be parallel.

    Returns:
        1D array:
            Areas of the parallelograms defined by vectors1 and vectors2
    """
    return np.sqrt(np.sum(np.cross(vectors1, vectors2) ** 2, axis=1))


def vectorized_triangle_area(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelogram.
        vectors2: 2D array
            Vectors where each row represents second side a paralellogram.

    Returns:
        1D array:
            Areas of the triangles defined by vectors1 and vectors2
    """
    return vectorized_parallelogram_area(vectors1, vectors2) * 0.5


def vectorized_parallelepiped_volume(vectors1, vectors2, vectors3):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelepiped.
        vectors2: 2D array
            Vectors where each row represents second side a parallelepiped.
        vectors3: 2D array
            Vectors where each row represents third side a parallelepiped.

    Returns:
        1D array:
            Volumes of the parallelepipeds defined by vectors1, vectors2 and vectors3
    """
    return np.abs(rowwise_dot(np.cross(vectors1, vectors2), vectors3))


def vectorized_tetrahedron_volume(vectors1, vectors2, vectors3):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a tetrahedron.
        vectors2: 2D array
            Vectors where each row represents second side a tetrahedron.
        vectors3: 2D array
            Vectors where each row represents third side a tetrahedron.

    Returns:
        1D array:
            Volumes of the tetrahedrons defined by vectors1, vectors2 and vectors3
    """
    return (1. / 6.) * vectorized_parallelepiped_volume(vectors1, vectors2, vectors3)


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


def subdivide_triangles(initial_points, initial_triangles, max_level=0, max_points=np.inf):
    """ Add face centers of triangles to the available points for the orientations.
    At each iteration all current triangles are substituted by their respective
    split triplets.

    Returns:
        added_points: array[float, (K, 3)]
            The new points that are created from the subdivisions.
        new_triangles: array[int, (M, 3)]
            The new split triangles from the subdivisions.
    """
    points = list(initial_points)  # copy
    new_triangles = list(initial_triangles)  # copy

    added_points = []

    offset = len(points)

    level = 0
    while level <= max_level and len(added_points) <= max_points:

        iteration_points = []
        iteration_triangles = []

        for i, triangle in enumerate(new_triangles):

            center = (points[triangle[0]] +
                      points[triangle[1]] +
                      points[triangle[2]]) / 3.0

            iteration_points.append(center)
            iteration_triangles.extend([
                (triangle[0], offset, triangle[1]),
                (triangle[1], offset, triangle[2]),
                (triangle[2], offset, triangle[0])
            ])

            offset += 1

            if len(added_points) + i + 1 >= max_points:
                break

        # the big triangles are substituted by
        # their children triplets
        new_triangles = iteration_triangles

        points.extend(iteration_points)
        added_points.extend(iteration_points)

        level += 1

    return added_points, new_triangles


def subdivide_triangles_by_total_area(points, triangles, min_number_of_points):
    """ Splits each triangle into three triangles with a new point at
    the center of the parent triangle. The total area of all triangles
    is used in order to determine the distribution of the number of
    points depending on the area of each triangle. Big triangles will
    have more points that small ones, according to the point area density
    calculated from the min_area_of_points.

    Given a number of points per triangle, the number of splitings has
    to be determined in order to achieve at least that number of points.
    N_level = 3 ^ L where N_level is the number of points on that level,
    and L the subsequent splitings.

    Thus, the series S[0, l](3^l) = 0.5(3^(l + 1) - 1) = N_total, where
    N_total is the total number of points after l splitings. Solving for
    the level:

    level = log3(2 * N_total + 1) - 1

    That is the maximum level that we reach by subsequent splitings in order
    to create N_total points.

    Args:
        points: array[float, (N, 3)]
        triangles: array[int, (M, 3)]
        target_number_of_points: int
            The minimum number of points to generate on the faces
            of the triangles.

    Returns:
        points: array[float, (N + K, 3)
            The initial array plus the added center points
        triangles: array[int, (W, 3)]
            A new set of subdivided triangles
    """
    points = np.asarray(points)
    triangles = np.asarray(triangles)

    if len(points) >= min_number_of_points:
        return points, triangles

    areas = vectorized_triangle_area(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]]
    )

    # don't take into account the existing points
    density = float(min_number_of_points - len(points)) / areas.sum()

    result_points = list(points)
    result_triangles = []

    sorted_ids = np.argsort(areas)[::-1]
    sorted_triangles = triangles[sorted_ids].tolist()
    points_per_triangle = np.rint(areas[sorted_ids] * density)

    for i, triangle in enumerate(sorted_triangles):

        n_points = points_per_triangle[i]

        try:
            max_level = \
                int(np.ceil(math.log(2. * n_points + 1., 3)) - 1.)
        except ValueError:
            continue

        new_points, new_triangles = subdivide_triangles(
            result_points,
            [triangle],
            max_level=max_level,
            max_points=n_points
        )

        result_points.extend(new_points)
        result_triangles.extend(new_triangles)

        if len(result_points) >= min_number_of_points:
            result_triangles.extend(sorted_triangles[i + 1::])
            break

    return np.asarray(result_points, dtype=np.float), \
           np.asarray(result_triangles, dtype=np.int)
