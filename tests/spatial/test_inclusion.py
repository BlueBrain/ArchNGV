
import numpy
from archngv.utils.linear_algebra import rowwise_dot

from archngv.spatial import inclusion as _inc


def create_spheres(center, radius, epsilon):

    directions = numpy.random.rand(5, 3)
    directions /= numpy.linalg.norm(directions, axis=1)[:, numpy.newaxis]

    radii = numpy.random.uniform(6., 8., size=5)

    centers = center + directions * (radius + radii + epsilon)[:, numpy.newaxis]

    return centers, radii


def test_spheres_in_sphere_inside():

    center = numpy.random.random(3)
    radius = numpy.random.uniform(10., 15.)

    directions = numpy.random.rand(5, 3)
    directions /= numpy.linalg.norm(directions, axis=1)[:, numpy.newaxis]

    radii = numpy.random.uniform(1., 2., size=5)

    centers = center + directions * (radius - radii - 1.)[:, numpy.newaxis]

    are_inside = _inc.spheres_in_sphere(centers, radii, center, radius)

    assert numpy.all(are_inside)


def test_sphere_in_sphere_outside():

    center = numpy.random.random(3)
    radius = numpy.random.uniform(10., 15.)

    # slightly outside sphere
    centers, radii = create_spheres(center, radius, 0.1)

    are_inside = _inc.spheres_in_sphere(centers, radii, center, radius)

    assert numpy.all(~are_inside)


def test_sphere_in_sphere_touching_outside():

    center = numpy.random.random(3)
    radius = numpy.random.uniform(10., 15.)

    # touching outside
    centers, radii = create_spheres(center, radius, 0.)

    are_inside = _inc.spheres_in_sphere(centers, radii, center, radius)

    assert numpy.all(~are_inside)


def test_sphere_in_sphere_touching_inside():

    center = numpy.random.random(3)
    radius = numpy.random.uniform(10., 15.)

    directions = numpy.random.rand(5, 3)
    directions /= numpy.linalg.norm(directions, axis=1)[:, numpy.newaxis]

    radii = numpy.random.uniform(6., 8., size=5)

    centers = center + directions * (radius - radii)[:, numpy.newaxis]

    are_inside = _inc.spheres_in_sphere(centers, radii, center, radius)

    assert numpy.all(are_inside)
