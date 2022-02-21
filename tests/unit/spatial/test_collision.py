import numpy as np

import archngv.utils.geometry as _gm
import archngv.utils.linear_algebra as _la
import archngv.utils.projections as _pj
from archngv.spatial import collision
from archngv.spatial.utils import create_contact_sphere_around_truncated_cylinder


def _format_sphere_spheres(center, radius, centers, radii):
    p = ["{} {}".format(str(c), str(r)) for c, r in zip(centers, radii)]
    return "\nSphere:\n {} {}\nSphers\n{}".format(str(center), str(radius), "\n".join(p))


def test_sphere_with_spheres_not_intersecting():

    center = np.random.random(3) * 31.0
    radius = np.random.uniform(1.0, 5.0)

    u_v = np.asarray(_gm.uniform_cartesian_unit_vectors(10)).T

    spheres_radii = np.random.uniform(2, 4, size=10)

    spheres_centers = center + (spheres_radii[:, np.newaxis] + 10.0) * u_v

    bool_arr = collision.sphere_with_spheres(center, radius, spheres_centers, spheres_radii)

    assert np.all(~bool_arr), _format_sphere_spheres(center, radius, spheres_centers, spheres_radii)


def test_sphere_with_spheres_intersecting():

    center = np.random.random(3) * 31.0
    radius = np.random.uniform(1.0, 10.0)

    u_v = np.asarray(_gm.uniform_cartesian_unit_vectors(10)).T

    r = np.sqrt(np.random.random(size=10)) * (10.0 * radius - radius) + radius

    spheres_centers = center + r[:, np.newaxis] * u_v
    spheres_radii = np.linalg.norm(spheres_centers - center, axis=1) - 0.5 * radius

    bool_arr = collision.sphere_with_spheres(center, radius, spheres_centers, spheres_radii)

    assert np.all(bool_arr), _format_sphere_spheres(center, radius, spheres_centers, spheres_radii)


def test_sphere_with_capsules_not_intersecting():

    p0s = np.array([[1.0, 1.0, 1.0], [10.0, 20.0, 10.0]])
    p1s = np.array([[10.0, 10.0, 10.0], [1.0, 11.0, 1.0]])

    r0s = np.array([3.0, 3.0])
    r1s = np.array([3.0, 1.0])

    x, y, z, r = create_contact_sphere_around_truncated_cylinder(p0s[0], p1s[0], r0s[0], r1s[0])

    t = collision.sphere_with_capsules(np.array((x[0], y[0], z[0])), r[0], p0s, p1s, r0s, r1s)

    assert np.all(~t)


def test_sphere_with_capsules_intersecting():

    p0s = np.array([[1.0, 1.0, 1.0], [10.0, 20.0, 10.0]])
    p1s = np.array([[10.0, 10.0, 10.0], [1.0, 11.0, 1.0]])

    r0s = np.array([3.0, 3.0])
    r1s = np.array([3.0, 1.0])

    x, y, z, r = create_contact_sphere_around_truncated_cylinder(p0s[0], p1s[0], r0s[0], r1s[0])

    t = collision.sphere_with_capsules(
        np.array((x[0], y[0], z[0])), r[0] * 10.0, p0s, p1s, r0s, r1s
    )

    assert np.all(t)


def test_convex_shape_with_spheres_touching_outwards():

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    face_vertices = np.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = np.array(
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.57735027, 0.57735027, 0.57735027]]
    )

    centroid = np.mean(points, axis=0)
    vectors = points[face_vertices[:, 0]] - centroid

    spheres_radii = np.random.uniform(0.2, 0.4, size=4)

    offs = _pj.rowwise_scalar_projections(vectors, face_normals)

    spheres_centers = centroid + face_normals * (offs + spheres_radii)[:, np.newaxis]

    checks = collision.convex_shape_with_spheres(
        points, face_normals, spheres_centers, spheres_radii
    )

    assert np.all(checks), (spheres_centers, spheres_radii)

    spheres_centers = centroid + face_normals * (offs + spheres_radii + 1.0)[:, np.newaxis]

    checks = collision.convex_shape_with_spheres(
        points, face_normals, spheres_centers, spheres_radii
    )

    assert np.all(~checks), (spheres_centers, spheres_radii)


def test_convex_shape_with_point():

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    face_vertices = np.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = np.array(
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.57735027, 0.57735027, 0.57735027]]
    )

    sample = np.random.uniform(low=(0.01, 0.01, 0.01), high=(0.2, 0.2, 0.2))

    check_func = lambda point: collision.convex_shape_with_point(
        points[face_vertices[:, 0]], face_normals, point
    )

    for point in sample:
        assert check_func(point)

    sample = np.random.uniform(low=(-1.0, -1.0, -1.0), high=(-0.1, -0.1, -0.1))

    for point in sample:
        assert not check_func(point)

    for point in points:
        assert check_func(point)
