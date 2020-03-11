""" Functions for the calculation of surface endfeet targets from their graph skeleton targets
and the somata positions.
"""
import math
import numpy as np
from archngv_building.endfeet_reconstruction.local_solvers import second_order_solutions


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


def surface_intersect(astrocyte_positions, potential_targets, astrocyte_target_edges, vasculature):
    """ From the line segments starting from target points on the skeleton of the vasculature
    graph and ending to the astrocytic somata the intersection with the surface of teh cones or
    cyliners is calculated.

    Args:
        astrocyte_positions: array[float, (N, 3)]
            Positions of the astrocytic somata.
        potential_targets: Dataframe of length M
            The potential targets to connect to. The following columns are required:
            x, y, z, edge_index
            where edge_id is the edge index of the vasculature the target is located
        astrocyte_target_edges: array[int, (K)]
            The edges between astrocyte somata and targets. Note that K < M
        vasculature: Vasculature
            The vasculature geometry/topology

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

    astrocyte_positions = astrocyte_positions.astype(np.float32)

    segments_start, segments_end = vasculature.segment_points.astype(np.float32)
    sg_radii_start, sg_radii_end = vasculature.segment_radii.astype(np.float32)
    edges = vasculature.edges.astype(np.int64)
    graph = vasculature.point_graph

    somata_idx, target_idx = astrocyte_target_edges.T.astype(np.int64)

    # get target properties
    target_edge_indices = potential_targets.loc[target_idx, 'edge_index'].to_numpy(dtype=np.int64)
    target_positions = potential_targets.loc[target_idx, ['x', 'y', 'z']].to_numpy(dtype=np.float32)

    n_connections = len(astrocyte_target_edges)
    surface_target_positions = np.empty((n_connections, 3), dtype=np.float32)
    surface_astros_idx = np.empty(n_connections, dtype=np.int64)
    vasculature_edge_idx = np.empty_like(surface_astros_idx)

    n_established = 0
    for astro_id, edge_id, target_point in zip(somata_idx, target_edge_indices, target_positions):

        target_to_soma_vec = astrocyte_positions[astro_id] - target_point

        for _ in range(10):
            # we have to determine if the intersection is with the current
            # segment or with a neighboring one. Therefore the intersection
            # is set to not_resolved until the correct segment is found or
            # the astro-target segment is contained in a truncated cone

            segment_beg_point = segments_start[edge_id]
            segment_end_point = segments_end[edge_id]

            segment_beg_radius = sg_radii_start[edge_id]
            segment_end_radius = sg_radii_end[edge_id]

            segment_length = np.linalg.norm(segment_beg_point - segment_end_point)

            if abs(segment_beg_radius - segment_end_radius) < EPS:  # cylinder

                # unit length direction of the cone
                cone_direction = (segment_end_point - segment_beg_point) / segment_length
                roots = cylinder_intersections(
                    cone_direction, segment_beg_point, target_point, target_to_soma_vec, segment_end_radius ** 2)

            else:  # truncated cone

                # make sure that the vector is always from the small radius to the big one
                (
                    segment_beg_point, segment_end_point,
                    segment_beg_radius, segment_end_radius
                ) = _resolve_segment_direction(
                    segment_beg_point, segment_end_point,
                    segment_beg_radius, segment_end_radius)

                # unit length direction of the cone
                cone_direction = (segment_end_point - segment_beg_point) / segment_length

                # opening angle of the truncated cone
                cone_angle = opening_angle(segment_beg_radius, segment_end_radius, segment_length)

                # coordinates of the tip of the truncated cone
                cone_tip = segment_beg_point - truncated_length(
                    segment_beg_radius, segment_end_radius, segment_length) * cone_direction

                # target - astro segment angle with the line of the segment
                # min and max angles determined from the size of the caps
                roots = cone_intersections(
                    cone_direction, cone_tip, target_point, target_to_soma_vec, cone_angle)

            # segment extent validity
            if -T_EPS < roots[0] < 1. + T_EPS:
                fraction = roots[0]
            elif -T_EPS < roots[1] < 1. + T_EPS:
                fraction = roots[1]
            else:
                # if the roots are not in [0, 1] then they aren't valid to proceed
                break

            surface_point = target_point + target_to_soma_vec * fraction
            left = np.dot(cone_direction, surface_point - segment_beg_point)
            right = np.dot(cone_direction, surface_point - segment_end_point)

            # after determining the point on the surface
            # validate its inclusion in the finite geometry
            if left > 0. > right:
                surface_astros_idx[n_established] = astro_id
                vasculature_edge_idx[n_established] = edge_id
                surface_target_positions[n_established] = surface_point
                n_established += 1
                break

            if left < 0. and right < 0.:  # check previous segment

                cid = edges[edge_id][0]
                # the int casting is a fix because np.uin64 is converted
                # to float if added to a regular int
                parents = graph.predecessors(cid)

                if parents.size > 0:
                    edge_id = graph.edge_index(parents[0], cid)
                else:  # if there is no parent
                    break

            elif left > 0. and right > 0.:  # check next segment

                pid = edges[edge_id][1]
                children = graph.successors(pid)
                if children.size > 0:
                    edge_id = graph.edge_index(pid, children[0])
                else:  # if there are no children
                    break
            else:
                break

    return (surface_target_positions[:n_established],
            surface_astros_idx[:n_established],
            vasculature_edge_idx[:n_established])
