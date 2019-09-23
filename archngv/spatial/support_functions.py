""" Support functions
"""
import numpy

from archngv.utils.linear_algebra import vectorized_dot_product


def sphere(center, radius, normalized_direction):
    """ Returns support for sphere along the normalized_direction
    """
    return center + normalized_direction * radius


def convex_polytope(shape_points, adjacency, direction):  # pylint: disable = W0613
    """ Returns the support of a convex_polytope along direction
    """
    products = vectorized_dot_product(shape_points, direction)
    return shape_points[numpy.argmax(products)]


def convex_polytope2(shape_points, adjacency, direction):
    """ Returns the support of a convex_polytope along direction
    """

    cur_vid = 0
    cur_dst = (shape_points[cur_vid] * direction).sum()

    while 1:

        new_vid = cur_vid
        new_dst = cur_dst

        for vid in adjacency[cur_vid]:

            dst = (shape_points[vid] * direction).sum()

            if dst > new_dst:

                new_vid = vid
                new_dst = dst

        if new_vid == cur_vid:
            break

        cur_vid = new_vid
        cur_dst = new_dst

    return shape_points[cur_vid]


def cylinder(centroid, central_axis, radius, height, direction):
    """ Returns the suppport of cylinder along direction
    """
    dot_u_d = numpy.dot(central_axis, direction)
    proj_u_d = dot_u_d * central_axis
    orth_component = direction - proj_u_d

    half_height = 0.5 * height

    if numpy.allclose(orth_component, 0.):

        return centroid + numpy.sign(dot_u_d) * half_height * central_axis

    return centroid + \
           numpy.sign(dot_u_d) * half_height * central_axis + \
           radius * orth_component / numpy.linalg.norm(orth_component)
