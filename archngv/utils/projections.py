"""
Functions related to projections
"""
import numpy as np

from archngv.utils.linear_algebra import rowwise_dot
from archngv.utils.linear_algebra import normalize_vectors


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
