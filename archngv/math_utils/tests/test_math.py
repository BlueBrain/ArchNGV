
import numpy
import numpy as np
import archngv.math_utils as _mt


def test_skew_symmetric_matrix():

    v = numpy.array((1., 2., 3.))

    A = _mt.skew_symmetric_matrix(v)

    result = numpy.array(((0., -3., 2.), (3., 0., -1.), (-2., 1., 0.)))

    assert numpy.allclose(A, result)


def test_angle_between_vectors():

    u = numpy.array((2., 1., 3.))
    v = numpy.array((-1., 5., 4.))

    angle = _mt.angle_between_vectors(u, v)

    assert numpy.isclose(angle, 0.90384998229123237)


def test_normalize_vectors():

    vec = numpy.random.rand(4, 3)

    expected_result = vec / numpy.linalg.norm(vec, axis=1)[:, numpy.newaxis]

    vectors = _mt.normalize_vectors(vec)

    assert numpy.allclose(vectors, expected_result)


def test_vectorized_dot_product():

    vec1 = numpy.random.rand(3)
    vecs = numpy.random.rand(4, 3)

    expected_result = numpy.fromiter((numpy.dot(vec1, v) for v in vecs), dtype=numpy.float)

    result = _mt.vectorized_dot_product(vecs, vec1)

    assert numpy.allclose(expected_result, result), '\n{}\n{}'.format(expected_result, result)


def test_vectorized_parallelepiped_volume():

    vectors1 = numpy.array([(1., 1., 1.),
                         (5., 1., 3.)])
    vectors2 = numpy.array([(2., 1., 2.),
                         (4., 1., 2.)])
    vectors3 = numpy.array([(2., 4., 4.),
                         (2., 1., 4.)])

    res_arr = _mt.vectorized_parallelepiped_volume(vectors1, vectors2, vectors3)
    act_arr = numpy.array([2., 4.])

    assert numpy.allclose(res_arr, act_arr)


def test_vectorized_tetrahedron_volume():

    vectors1 = numpy.array([(1., 1., 1.),
                         (5., 1., 3.)])
    vectors2 = numpy.array([(2., 1., 2.),
                         (4., 1., 2.)])
    vectors3 = numpy.array([(2., 4., 4.),
                         (2., 1., 4.)])

    res_arr = _mt.vectorized_tetrahedron_volume(vectors1, vectors2, vectors3)
    act_arr = numpy.array([2., 4.]) / 6.

    assert numpy.allclose(res_arr, act_arr)


def test_cartesian_product():

    arr1 = numpy.array([4, 8])
    arr2 = numpy.array([1, 2])
    arr3 = numpy.array([0, 3])

    expected_result = \
        numpy.array([[4, 1, 0],
                     [4, 1, 3],
                     [4, 2, 0],
                     [4, 2, 3],
                     [8, 1, 0],
                     [8, 1, 3],
                     [8, 2, 0],
                     [8, 2, 3]], dtype=numpy.intp)

    result = _mt.cartesian_product(arr1, arr2, arr3)

    assert numpy.allclose(expected_result, result)


def test_scalar_projection():

    vec1 = numpy.random.rand(3)
    vecs = numpy.random.rand(4, 3)

    u_vec = vec1 / numpy.linalg.norm(vec1)

    expected_result = numpy.fromiter((numpy.dot(v, u_vec) for v in vecs), dtype=numpy.float)

    result = _mt.vectorized_scalar_projection(vecs, vec1)

    assert numpy.allclose(expected_result, result), '\n{}\n{}'.format(expected_result, result)


def test_scalar_projections():

    vecs1 = numpy.random.rand(4, 3)
    vecs2 = numpy.random.rand(4, 3)

    uvecs = numpy.linalg.norm(vecs2, axis=1)

    expected_result = \
        numpy.fromiter((numpy.dot(v1, v2) / u2 for v1, v2, u2 in zip(vecs1, vecs2, uvecs)), dtype=numpy.float)

    result = _mt.rowwise_scalar_projections(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


def test_vector_projection():

    vec1 = numpy.random.rand(3)
    vecs = numpy.random.rand(4, 3)

    u_vec = vec1 / numpy.linalg.norm(vec1)

    expected_result = numpy.vstack([numpy.dot(v, u_vec) * u_vec for v in vecs])

    result = _mt.vectorized_vector_projection(vecs, vec1)

    assert numpy.allclose(expected_result, result)


def test_vectors_projections():

    vecs1 = numpy.random.rand(4, 3)
    vecs2 = numpy.random.rand(4, 3)

    uvecs = numpy.linalg.norm(vecs2, axis=1)

    expected_result = numpy.vstack([numpy.dot(v1, v2) * v2 / u2 ** 2 for v1, v2, u2 in zip(vecs1, vecs2, uvecs)])

    result = _mt.rowwise_vector_projections(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


def test_projection_vector_on_plane():

    vectors = numpy.random.rand(4, 3)

    normal = numpy.array([1., 0., 0.])

    expected_result = numpy.vstack([v - numpy.dot(normal, v) * normal for v in vectors])

    result = _mt.vectorized_projection_vector_on_plane(vectors, normal)

    assert numpy.allclose(expected_result, result)


def test_triangle_normal():

    As = numpy.random.rand(2, 3)
    Bs = numpy.random.rand(2, 3)
    Cs = numpy.random.rand(2, 3)

    Us = Bs - As
    Vs = Cs - As

    result = []

    for i in range(2):

        x = (Us[i][1] * Vs[i][2]) - (Us[i][2] * Vs[i][1])
        y = (Us[i][2] * Vs[i][0]) - (Us[i][0] * Vs[i][2])
        z = (Us[i][0] * Vs[i][1]) - (Us[i][1] * Vs[i][0])
        dist = numpy.sqrt(x**2 + y**2 + z**2)
        n = numpy.asarray((x / dist, y / dist, z / dist))
        result.append(n)

    expected_result = numpy.asarray(result)

    result = _mt.vectorized_triangle_normal(Bs - As, Cs - As)

    assert numpy.allclose(expected_result, result)


def test_rowwise_dot():

    vecs1 = numpy.random.rand(5, 4)
    vecs2 = numpy.random.rand(5, 4)

    expected_result = numpy.fromiter((numpy.dot(v1, v2) for v1, v2 in zip(vecs1, vecs2)), dtype=numpy.float)

    result = _mt.rowwise_dot(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


def test_are_in_the_same_side():

    vecs1 = numpy.array([[1., 2., 1.],
                         [-1., -1., -1.],
                         [1., 1., 1.]])

    vecs2 = numpy.array([[0., 1., 1.],
                         [1., 2., 1.],
                         [1., 1., 1.]])

    expected_result = numpy.array([True, False, True], dtype=numpy.bool)

    result = _mt.are_in_the_same_side(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


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

    result = _mt.angle_matrix(a, b)

    assert result.shape == expected_shape
    assert np.allclose(result, expected_matrix)


