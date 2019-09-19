""" GJK algorithm
https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm
"""
import numpy as np

# pylint: disable = C0103, R0913, R0914, R0915, invalid-unary-operand-type


def dot(a, b):
    """ Dot product between two vectors """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    """ Cross product between two vectors """
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


def support(shape1, shape2, direction):
    """ Support function to get the Minkowsky difference
    """
    direction = direction / np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    return shape1.support(direction) - shape2.support(-direction)


def pick_line(shape1, shape2, direction):
    """ Pick line"""
    return (support(shape2, shape1, direction),
            support(shape2, shape1, -direction))


def pick_triangle(a, b, shape1, shape2, Niter):
    """ Pick triangle
    """
    flag = 0

    ab = b - a
    ao = -a

    v = cross(cross(ab, ao), ab)

    c = b
    b = a
    a = support(shape2, shape1, v)

    for _ in range(Niter):

        ab = b - a
        ao = -a
        ac = c - a

        abc = cross(ab, ac)

        abp = cross(ab, abc)

        acp = cross(abc, ac)

        if dot(abp, ao) > 0.:

            c = b
            b = a
            v = abp

        elif dot(acp, ao) > 0.:

            b = a
            v = acp

        else:

            flag = 1
            break

        a = support(shape2, shape1, v)

    return a, b, c, flag


def pick_tetrahedron(a, b, c, shape1, shape2, Niter):
    """ pick tetrahedron
    """
    flag = 0

    ab = b - a
    ac = c - a

    abc = cross(ab, ac)

    ao = -a

    if dot(abc, ao) > 0:

        d = c
        c = b
        b = a

        v = abc
        a = support(shape2, shape1, v)

    else:

        d = b
        b = a
        v = -abc
        a = support(shape2, shape1, v)

    for _ in range(Niter):

        ab = b - a
        ao = -a
        ac = c - a
        ad = d - a

        abc = cross(ab, ac)

        if dot(abc, ao) <= 0.:

            acd = cross(ac, ad)

            if dot(acd, ao) > 0.:

                b = c
                c = d
                ab = ac
                ac = ad
                abc = acd

            else:

                adb = cross(ad, ab)

                if dot(adb, ao) > 0.:

                    c = b
                    b = d
                    ac = ab
                    ab = ad
                    abc = adb

                else:

                    flag = 1
                    break

        if dot(abc, ao) > 0:

            d = c
            c = b
            b = a
            v = abc
            a = support(shape2, shape1, v)

        else:

            d = b
            b = a
            v = -abc
            a = support(shape2, shape1, v)

    return a, b, c, d, flag


def GJK(shape1, shape2, max_iter):
    """ GJK algorithm for two shapes
    """
    initial_direction = shape2.centroid - shape1.centroid

    if initial_direction[0] == initial_direction[1] == initial_direction[2] == 0.:
        initial_direction = np.array([1., 0., 0.])

    a, b = pick_line(shape2, shape1, initial_direction)

    a, b, c, found_triangle = pick_triangle(a, b, shape2, shape1, max_iter)

    if found_triangle:

        a, b, c, _, have_collided = pick_tetrahedron(a, b, c, shape2, shape1, max_iter)

    else:

        have_collided = False

    return have_collided
