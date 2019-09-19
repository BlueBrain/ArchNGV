""" Functions related to linear algebra
"""

import numpy as np


def normalize_vectors(vectors):
    """
    Args:
        vectors: array_like

    Returns:
        Rowwise normalised vectors
    """
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, np.newaxis]


def vectorized_dot_product(vectors, vector):
    """
    Args:
        vectors: array_like
            2D array with row vectors
        vector: array_like
            1D array

    Returns:
        Dot products of vector with each row of vectors
    """
    return np.einsum('i,ji->j', vector, vectors)


def vectorized_scalar_projection(vectors, vector):
    """ Projects and array of vectors onto another vector
    """
    return np.inner(vectors, vector) / np.linalg.norm(vector)


def vectorized_vector_projection(vectors, vector):
    """
    Args:
        vectors: array_like
            2D array with row vectors
        vector: array_like
            1D array

    Returns:
        2D array
            Rowwise vector projections of vectors onto vector
    """
    sc_proj = vectorized_scalar_projection(vectors, vector)
    return np.multiply(sc_proj[np.newaxis].T, vector / np.linalg.norm(vector))


def rowwise_dot(vectors1, vectors2):
    """
    Args:
        vectors1: array_like
            2D array with row vectors
        vector2: array_like
            2D array with row vectors

    Returns:
        2D array
            Rowwise dot products
    """
    return np.sum(vectors1 * vectors2, axis=1)


def rowwise_scalar_projections(vectors1, vectors2):
    """
    Args:
        vectors1: array_like
            2D array with row vectors
        vector2: array_like
            2D array with row vectors

    Returns:
        1D array
            Rowwise scalar projections of each vector in vectors1
            onto the corresponding row vector in vectors2
    """
    u_vectors2 = normalize_vectors(vectors2)
    return rowwise_dot(vectors1, u_vectors2)


def rowwise_vector_projections(vectors1, vectors2):
    """
    Args:
        vectors1: array_like
            2D array with row vectors
        vector2: array_like
            2D array with row vectors

    Returns:
        2D array:
            Rowwise scalar projections of each vector in vectors1
            onto the corresponding row vector in vectors2
    """
    sc_projs = rowwise_scalar_projections(vectors1, vectors2)
    u_vectors2 = normalize_vectors(vectors2)
    return sc_projs[:, np.newaxis] * u_vectors2


def vectorized_projection_vector_on_plane(vectors, plane_normal):
    """
    Args:
        vectors: array_like
            2D array with row vectors
        plane_normal: array_like
            1D array of plane normal direction

    Returns:
        2D array:
            Rowwise vector projections of each vector in vectors
            to the plane defined by the normal
    """
    return vectors - vectorized_vector_projection(vectors, plane_normal)


def vectorized_projection_point_on_plane(points, plane_point, plane_normal):
    """
    Args:
        points: array_like
            2D array of 3D points
        plane_point: array_like
            1D array of a point on the plane
        plane_normal: array_like
            1D array of the plane normal

    Returns:
        2D array:
            Rowwise projection of points onto the plane defined by the plane_point
            and plane_normal
    """
    vectors = plane_point - points
    projs = vectorized_projection_vector_on_plane(vectors, plane_normal)
    return plane_point + projs


def skew_symmetric_matrix(vec):
    """
    Args:
        vec: array_like
            1D array of vector the skew symmetric matrix will be constructed from
    Returns:
        2D array:
            A 3x3 skew-symmetric matrix from vector v
    """
    return np.array(((0., - vec[2], vec[1]), (vec[2], 0., -vec[0]), (- vec[1], vec[0], 0.)))


def angle_between_vectors(vector_1, vector_2):
    """
    Args:
        vector_1, vector_2: array_like
            1D vectors

    Returns:
        float:
            The angle between vectors u and v in radians.
    """
    c_p = np.cross(vector_1, vector_2)
    return np.arctan2(np.linalg.norm(c_p), np.dot(vector_1, vector_2))


def principal_directions(points, return_eigenvalues=False):
    """
    Args:
        points: array_like
            2D array of per-row points

    Returns:
        The eigenvalues and eigenvectors of the covariance matrix
        created from the set of points sorted from biggest to smallest
    """
    covariance = np.cov(points.T)

    eigs, eigv = np.linalg.eig(covariance)

    idx = np.argsort(eigs)[::-1]

    if return_eigenvalues:

        return eigs[idx], eigv[:, idx].T

    return eigv[:, idx].T


def are_in_the_same_side(vectors1, vectors2):
    """
    Args:
        vectors1: array_like
            2D array with row vectors
        vector2: array_like
            2D array with row vectors

    Returns:
        Bool: 1D array
            Checks whether each vector in vectors1 is pointing
            at the same halfspace as the corresponding vector
            in vectors2.
    """
    return rowwise_dot(vectors1, vectors2) > 0.


def angle_matrix(vectors1, vectors2):
    """ Calculate all the pairwise angles between vectors1 and vectors2
    Args:
        vectors1: float[array, (N, 3)]
        vectors2: float[array, (M, 3)]
    Returns: array[float, (N, M)]
    """
    dot_matrix = np.inner(
        normalize_vectors(vectors1),
        normalize_vectors(vectors2)
    )
    return np.arccos(np.clip(dot_matrix, -1.0, 1.0))
