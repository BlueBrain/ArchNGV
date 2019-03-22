
def equal_length(iterable1, iterable2):

    try:

        l1 = len(iterable1)
        l2 = len(iterable2)

        assert l1 == l2

    except AssertionError:

        msg = 'Iterables do not have same length {}'.format(l1, l2)
        L.error(msg)
        raise AssertionError(msg)


def keys(keys_to_check, dictionary):

    try:

        for key in keys_to_check:
            assert key in dictionary

    except AssertionError:

        msg = '{} key could not be found in config'.format(key)
        L.error(msg)
        raise AssertionError(msg)


def points_inside_polyhedra(points, polyhedra):
    from morphspatial.collision import convex_shape_with_point

    try:

        for point, polyhedron in zip(points, polyhedra):
            assert convex_shape_with_point(polyhedron.face_points,
                                           polyhedron.face_normals,
                                           point)

    except AssertionError:

        msg = 'Points are not inside polyhedra.'
        L.error(msg)
        raise AssertionError(msg)
