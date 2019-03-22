# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import time
import logging
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

from .linked_list cimport SinglyLinkedList

from cython.view cimport array as cvarray

L = logging.getLogger(__name__)

ctypedef np.npy_intp SIZE_t


cdef set find_surface_contours(dict edge_triangles):

    """
    Find the chain of indices that makes up the contour of the endfoot.
    """
    cdef:
        list contours
        tuple edge
        set edges, to_remove

        SinglyLinkedList chain

        SIZE_t v1, v2, head, tail


    # edges that belong to one triangle are exterior ones
    edges = set(tuple(edge) for edge, triangles in edge_triangles.items() if len(triangles) == 1)

    try:
        v1, v2 = edges.pop()
    except KeyError:
        L.warning('No boundary edges found.')
        return set()

    chain = SinglyLinkedList()
    chain.add_left(v1)
    chain.add_right(v2)

    contours = [chain]

    # exand chains left and right until we end up with single cyclic one
    while edges:

        to_remove = set()

        for edge in edges:

            v1, v2 = edge

            found = False

            for chain in contours:

                head = chain.head_value()
                tail = chain.tail_value()

                if head == v1:
                    chain.add_left(v2)
                    to_remove.add(edge)
                    break

                if head == v2:
                    chain.add_left(v1)
                    to_remove.add(edge)
                    break

                if tail == v1:
                    chain.add_right(v2)
                    to_remove.add(edge)
                    break

                if tail == v2:
                    chain.add_right(v2)
                    to_remove.add(edge)
                    break

        if len(to_remove) > 0:

           edges -= to_remove

        else:

            (v1, v2) = edges.pop()

            chain = SinglyLinkedList()
            chain.add_left(v1)
            chain.add_right(v2)

            contours.append(chain)

    #assert chain[0] == chain[-1], 'Chain start not the same as chain end.'

    return set(v1 for chain in contours for v1 in chain.values())

