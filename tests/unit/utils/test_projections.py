import numpy

from archngv.utils import linear_algebra as la
from archngv.utils import projections as pj


def test_scalar_projection():
    vec1 = numpy.random.rand(3)
    vecs = numpy.random.rand(4, 3)

    u_vec = vec1 / numpy.linalg.norm(vec1)

    expected_result = numpy.fromiter((numpy.dot(v, u_vec) for v in vecs), dtype=float)

    result = pj.vectorized_scalar_projection(vecs, vec1)

    assert numpy.allclose(expected_result, result), "\n{}\n{}".format(expected_result, result)


def test_scalar_projections():
    vecs1 = numpy.random.rand(4, 3)
    vecs2 = numpy.random.rand(4, 3)

    uvecs = numpy.linalg.norm(vecs2, axis=1)

    expected_result = numpy.fromiter(
        (numpy.dot(v1, v2) / u2 for v1, v2, u2 in zip(vecs1, vecs2, uvecs)), dtype=float
    )

    result = pj.rowwise_scalar_projections(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


def test_vector_projection():
    vec1 = numpy.random.rand(3)
    vecs = numpy.random.rand(4, 3)

    u_vec = vec1 / numpy.linalg.norm(vec1)

    expected_result = numpy.vstack([numpy.dot(v, u_vec) * u_vec for v in vecs])

    result = pj.vectorized_vector_projection(vecs, vec1)

    assert numpy.allclose(expected_result, result)


def test_vectors_projections():
    vecs1 = numpy.random.rand(4, 3)
    vecs2 = numpy.random.rand(4, 3)

    uvecs = numpy.linalg.norm(vecs2, axis=1)

    expected_result = numpy.vstack(
        [numpy.dot(v1, v2) * v2 / u2**2 for v1, v2, u2 in zip(vecs1, vecs2, uvecs)]
    )

    result = pj.rowwise_vector_projections(vecs1, vecs2)

    assert numpy.allclose(expected_result, result)


def test_projection_vector_on_plane():
    vectors = numpy.random.rand(4, 3)

    normal = numpy.array([1.0, 0.0, 0.0])

    expected_result = numpy.vstack([v - numpy.dot(normal, v) * normal for v in vectors])

    result = pj.vectorized_projection_vector_on_plane(vectors, normal)

    assert numpy.allclose(expected_result, result)
