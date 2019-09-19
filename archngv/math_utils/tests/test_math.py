
import numpy as np
from archngv.math_utils import linear_algebra as _la


def test_skew_symmetric_matrix():

    v = np.array((1., 2., 3.))

    A = _la.skew_symmetric_matrix(v)

    result = np.array(((0., -3., 2.), (3., 0., -1.), (-2., 1., 0.)))

    assert np.allclose(A, result)


def test_angle_between_vectors():

    u = np.array((2., 1., 3.))
    v = np.array((-1., 5., 4.))

    angle = _la.angle_between_vectors(u, v)

    assert np.isclose(angle, 0.90384998229123237)


def test_normalize_vectors():

    vec = np.random.rand(4, 3)

    expected_result = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]

    vectors = _la.normalize_vectors(vec)

    assert np.allclose(vectors, expected_result)


def test_vectorized_dot_product():

    vec1 = np.random.rand(3)
    vecs = np.random.rand(4, 3)

    expected_result = np.fromiter((np.dot(vec1, v) for v in vecs), dtype=np.float)

    result = _la.vectorized_dot_product(vecs, vec1)

    assert np.allclose(expected_result, result), '\n{}\n{}'.format(expected_result, result)


def test_rowwise_dot():

    vecs1 = np.random.rand(5, 4)
    vecs2 = np.random.rand(5, 4)

    expected_result = np.fromiter((np.dot(v1, v2) for v1, v2 in zip(vecs1, vecs2)), dtype=np.float)

    result = _la.rowwise_dot(vecs1, vecs2)

    assert np.allclose(expected_result, result)


def test_are_in_the_same_side():

    vecs1 = np.array([[1., 2., 1.],
                         [-1., -1., -1.],
                         [1., 1., 1.]])

    vecs2 = np.array([[0., 1., 1.],
                         [1., 2., 1.],
                         [1., 1., 1.]])

    expected_result = np.array([True, False, True], dtype=np.bool)

    result = _la.are_in_the_same_side(vecs1, vecs2)

    assert np.allclose(expected_result, result)


def test_angle_matrix():

    a = np.array([[0.1, 0.2, 0.3],
                  [-2., 0.3, -22.],
                  [5., 6., 11.]])


    b = np.array([[2., 1., -1.],
                  [-23., 1.3, -12.]])

    expected_shape = (len(a), len(b))
    expected_matrix = np.zeros(expected_shape)

    for i, v1 in enumerate(a):
        l_a = np.linalg.norm(v1)
        for j, v2 in enumerate(b):
            l_b = np.linalg.norm(v2)
            dt = np.dot(v1, v2) / (l_a * l_b)
            expected_matrix[i, j] = np.arccos(np.clip(dt, -1., 1.))

    result = _la.angle_matrix(a, b)

    assert result.shape == expected_shape
    assert np.allclose(result, expected_matrix)


