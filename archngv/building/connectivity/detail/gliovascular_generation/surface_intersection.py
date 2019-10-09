""" Functions for the calculation of surface endfeet targets from their graph skeleton targets
and the somata positions.
"""
import math
import numpy as np
from archngv.building.endfeet_reconstruction.detail.local_solvers import second_order_solutions


EPS = 1e-6


def truncated_length(r, R, L):
    """ Returns the length from the tip of the cone to the small truncated side

    Args:
        r: float
            Radius of small side of the truncated cone.
        R: float
            Radius of the big side of the truncated cone
        L: float
            Length of the truncated (cut) cone's interior.

    Notes:
        The truncated cone should be oriented so that: r < R
        The total length of the cone is l_t = l + L, where l, L are the lengths
        outside and inside the truncated code respectively.
        Using triangle similarity it holds that l_t / R = l / r -> (l + L) / R = l / r
    """
    return L * r / (R - r)


def opening_angle(r, R, L):
    """ Returns the opening angle of the truncated cone.

    Args:
        r: float
            Radius of small side of the truncated cone.
        R: float
            Radius of the big side of the truncated cone
        L: float
            Length of the truncated (cut) cone's interior.

    Notes:
        The truncated cone should be oriented so that: r < R
        Knowing the length from the tip of the cone to the smaller truncated side and its length (r)
        we can calculate the angle using the arctangent.
    """
    return math.atan2(r, truncated_length(r, R, L))


def norm(V, W):
    """ Returns the length between point V and W.
    """
    return math.sqrt((V[0] - W[0]) ** 2 +
                     (V[1] - W[1]) ** 2 +
                     (V[2] - W[2]) ** 2)


def _M(D, open_angle):
    R = D[:, None] * D
    cos_angle2 = math.cos(open_angle) ** 2
    R[0, 0] -= cos_angle2
    R[1, 1] -= cos_angle2
    R[2, 2] -= cos_angle2
    return R


def cone_intersections(D, V, T, TC, open_angle):
    r"""
    Args:
        D: array[float, (3,)]
        V: array[float, (3,)]
        T: array[float, (3,)]
        TC: array[float, (3,)]
        opening_angle: float


    Returns:
        roots: array[float, (2,)]
            The parametric t solutions to the second order equation.

    Notes:
        https://mrl.nyu.edu/~dzorin/rend05/lecture2.pdf

     Soma center   C     /
                    \   /
                     \ /
                      P (We search for this point)
                     | \
                    /|  \
                   / |   \
                  /  |    \
      Cone tip  V ->-|-----T------
                  D        Edge target
    """
    M = _M(D, open_angle)
    VT = T - V

    c0 = np.dot(np.dot(VT, M), VT)
    c1 = 2. * np.dot(np.dot(TC, M), VT)
    c2 = np.dot(np.dot(TC, M), TC)

    return second_order_solutions(c2, c1, c0)


def cylinder_intersections(V, S, T, TC, R2):
    r"""
    Args:
        V: array[float, (3,)]
            The normalized direction of the cylinder.
        S: array[float, (3,)]
            The start point of the segment.
        T: array[float, (3,)]
            The target point on the edge that connects the start and end of the cylinder.
        TC: array[float, (3,)]
            The vector from the target to the soma center.
        R2: float
            The squared radius of the cylinder cap.

     Returns:
         roots: array[float, (2,)]
             The parametric t solutions to the second order equation.

     Notes:
                        C (Soma center)
                         \
                          \ (We search for this point)
                    P'|----P------------------|
                      |   / \                 |
                      |  /   \                |
                      | /     \               |
                      |/       \              |
    Segment Start    S|-> V     T             |E  Segment end
    Segment Direction |        Edge target    |
                      |                       |
                      |                       |
                      |--------------P--------|
                                      (second solution, not wanted)

    The magnitude of the projection SP' of the vector SP on the cap of the cylinder should always
    be equal to r^2.

    proj_[V](SP) = <V, SP> * V
                 = <V, P - S> * V

    SP' = SP -  proj_[V0](SP)
        = SP - <V, SP> * V
        = P - S - <V, P - S> * V

    <SP', SP'> = r^2 -> (P - S - <V, P - S> * V)^2 - r^2 = 0

    The parametric form of the line segment TC is P = T + t * TC. Therefore if we substitute:

    (T + t * TC - S - <V, T + t * TC - S> * V)^2 - r^2 = 0

    (ST + t * TC - <V, ST + t * TC> * V)^2 - r^2 = 0
    (ST - <V, ST> * V + (TC - <V, TC> * V) * t) ^ 2 - r^2 = 0

    Now let:

    A = TC - <V, TC> * V
    B = ST - <V, ST> * V

    <B + At, B + At> - r^2 = 0
    <B, B> + 2<A, B>t + <A, A>t^2 - r^2 = 0

    Which is a second order equation with:

    a = <A, A>
    b = 2<A, B>
    c = <B, B> - r^2
    """
    ST = T - S
    A = TC - V * np.dot(TC, V)
    B = ST - V * np.dot(ST, V)

    c0 = np.dot(B, B) - R2
    c1 = 2. * np.dot(A, B)
    c2 = np.dot(A, A)

    return second_order_solutions(c2, c1, c0)


def _resolve_segment_direction(start_pos, end_pos, start_rad, end_rad):
    """ Swaps the start and end of a truncated cone so that the small side comes always first.
    """
    if start_rad - end_rad > EPS:
        return end_pos, start_pos, end_rad, start_rad
    return start_pos, end_pos, start_rad, end_rad


# pylint: disable=too-many-arguments
def surface_intersect(somata_positions,
                      target_positions,
                      segments_start, segments_end,
                      sg_radii_start, sg_radii_end,
                      somata_idx,
                      target_idx,
                      segment_idx,
                      edges,
                      graph):
    """ From the line segments starting from target points on the skeleton of the vasculature
    graph and ending to the astrocytic somata the intersection with the surface of teh cones or
    cyliners is calculated.

    Args:
        somata_positions: array[float, (N, 3)]
            Positions of the astrocytic somata.
        target_positions: array[float, (N, 3)]
            The points on the skeleton graph of the vasculature that connect to each soma.
        segments_starts: array[float, (M, 3)]
            The start points of the vasculature segments.
        segments_end: array[float, (M, 3)]
            The end points of the vasculature segments.
        sg_radii_start: array[float, (M,)]
            The start radii of the vasculature segments.
        sg_radii_end: array[float, (M,)]
            The end radii of the vasculature segments.
        somata_idx: array[int, (N,)]
            The ids for each astrocytic soma.
        target_idx: array[int, (N,)]
            The ids for each endfoot target on the skeleton.
        segment_idx: array[int]
            The ids for each edge.
        edges: array[int, (M, 2)]
            The edges of the vasculature graph.
        graph: Graph
            The point graph of the vasculature.

    Returns:
       surface_target_positions: array[float, (N, 3)]
           The points on the surface of the parametric cylinders / cones of the vasculature.
       surface_astros_idx: array[int, (N,)]
           The respective astrocyte index for each intersection.
       vasculature_edge_idx: array[int, (N,)]
           The respective vasculature edge index for each intersecion.
    """
    #  pylint: disable=too-many-locals,too-many-branches,too-many-statements
    T_EPS = 1e-1  # margin of error in the parametric t e.g 0.1 of length of segment

    surface_target_positions, surface_astros_idx, vasculature_edge_idx = [], [], []
    for astro_id, target_it in zip(somata_idx, target_idx):
        T0 = target_positions[target_it]
        U = somata_positions[astro_id] - T0

        # first segment id is the one that contains the target point
        sid = segment_idx[target_it]
        for _ in range(10):
            # we have to determine if the intersection is with the current
            # segment or with a neighboring one. Therefore the intersection
            # is set to not_resolved until the correct segment is found or
            # the astro-target segment is contained in a truncated cone

            start, end = segments_start[sid], segments_end[sid]

            radius_start, radius_end = sg_radii_start[sid], sg_radii_end[sid]

            length = norm(start, end)

            if abs(radius_start - radius_end) < EPS:  # cylinder
                # unit length direction of the cone
                D = (end - start) / length
                roots = cylinder_intersections(D, start, T0, U, radius_end ** 2)
            else:  # truncated cone
                # make sure that the vector is always from the small radius to the big one
                start, end, radius_start, radius_end = _resolve_segment_direction(
                    start, end, radius_start, radius_end)

                # unit length direction of the cone
                D = (end - start) / length
                cone_angle = opening_angle(radius_start, radius_end, length)

                # coordinates of the tip of the cone
                V = start - truncated_length(radius_start, radius_end, length) * D

                # target - astro segment angle with the line of the segment
                # min and max angles determined from the size of the caps
                roots = cone_intersections(D, V, T0, U, cone_angle)

            # segment extent validity
            if -T_EPS < roots[0] < 1. + T_EPS:
                T = roots[0]
            elif -T_EPS < roots[1] < 1. + T_EPS:
                T = roots[1]
            else:
                break

            P = T0 + U * T
            left, right = np.dot(D, P - start), np.dot(D, P - end)

            # after determining the point on the surface
            # validate its inclusion in the finite geometry
            if left > 0. > right:
                surface_target_positions.append(P)
                surface_astros_idx.append(astro_id)
                vasculature_edge_idx.append(sid)
                break

            if left < 0. and right < 0.:  # check previous segment
                cid = int(edges[sid][0])
                pids = graph.adjacency_matrix.parents(cid)

                if pids.size:
                    sid = graph.get_edge_index(pids[0], cid)
                else:  # if there is no parent
                    break
            elif left > 0. and right > 0.:  # check next segment
                pid = int(edges[sid][1])
                cids = graph.adjacency_matrix.children(pid)

                if cids.size:
                    sid = graph.get_edge_index(pid, cids[0])
                else:  # if there are no children
                    break
            else:
                break

    return (np.array(surface_target_positions),
            np.array(surface_astros_idx),
            np.array(vasculature_edge_idx),
            )
