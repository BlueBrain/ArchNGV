from libc.math cimport atan2, sqrt, cos, fabs
cimport numpy as np
import numpy as np

DEF EPS = 1e-6
DEF T_EPS = 1e-1 # margin of error in the parametric t e.g 0.1 of length of segment


cdef inline mult(double[:] V1, double[:] V2):

    cdef double[:] W = np.empty(3, dtype=np.float64)

    W[0] = V1[0] * V2[0]
    W[1] = V1[1] * V2[1]
    W[2] = V1[2] * V2[2]

    return W

cdef inline const_mult(double[:] V, double a):

    cdef double[:] W = np.empty(3, dtype=np.float64)

    W[0] = V[0] * a
    W[1] = V[1] * a
    W[2] = V[2] * a

    return W

cdef inline diff(double[:] V1, double[:] V2):

    cdef double[:] W = np.empty(3, dtype=np.float64)

    W[0] = V1[0] - V2[0]
    W[1] = V1[1] - V2[1]
    W[2] = V1[2] - V2[2]

    return W

cdef inline double truncated_length(double r, double R, double L):
    return  L * r / (R - r)


cdef inline double[:] tip_of_truncated_cone(double[:] D, double[:] P1, double h):

    cdef double[:] V = np.empty(3, dtype=np.float64)

    V[0] = P1[0] -  h * D[0]
    V[1] = P1[1] -  h * D[1]
    V[2] = P1[2] -  h * D[2]

    return V


cdef inline double opening_angle(double r, double R, double L):
    return atan2(r, truncated_length(r, R, L))


cdef inline double[:] parametric_point(double[:] P, double[:] U, double t):

    cdef double[:] T = np.empty(3, dtype=np.float64)

    T[0] = P[0] + U[0] * t
    T[1] = P[1] + U[1] * t
    T[2] = P[2] + U[2] * t

    return T

cdef inline double norm(double[:] V, double[:] W):
    return sqrt((V[0] - W[0]) ** 2 + (V[1] - W[1]) ** 2 + (V[2] - W[2]) ** 2)


cpdef inline double dot(double[:] V, double[:] W):
    return V[0] * W[0] + V[1] * W[1] + V[2] * W[2]


cdef inline double[:, :] _M(double[:] D, double opening_angle):

    cdef:
        double[:, :] R = np.empty((3, 3), dtype=np.float64)

        double cos_angle2 = cos(opening_angle) ** 2

    R[0, 0] = D[0] * D[0] - cos_angle2
    R[0, 1] = D[0] * D[1]
    R[0, 2] = D[0] * D[2]
    R[1, 1] = D[1] * D[1] - cos_angle2
    R[1, 2] = D[1] * D[2]
    R[2, 2] = D[2] * D[2] - cos_angle2

    R[1, 0] = R[0, 1]
    R[2, 0] = R[0, 2]
    R[2, 1] = R[1, 2]

    return R

cdef inline double[:] _second_order_solutions(double a, double b, double c):

    cdef:
        double[:] t = np.empty(2, dtype=np.float64)
        double Q

    Q = b ** 2 - 4. * a * c

    b = 0. if fabs(b) < EPS else b

    if fabs(Q) > EPS:

        if b >= 0.:

            t[0] = 0.5 * (- b - sqrt(Q)) / a
            t[1] = 2. * c / (- b - sqrt(Q))

        else:

            t[0] = 0.5 * (- b + sqrt(Q)) / a
            t[1] = 2. * c / (- b + sqrt(Q))

    elif fabs(Q) <= EPS:

        t[0] = - c / (2. * a)
        t[1] = - 1.

    else:

        t[0] = - 1.
        t[1] = - 1.

    return t


cdef double[:] cone_intersections(double[:] D,
                             double[:] V,
                             double[:] P,
                             double[:] U,
                             double opening_angle):

    cdef:
        double[:] W = np.empty(3, dtype=np.float64)
        double[:, :]  M = np.empty((3, 3), dtype=np.float64)
        double c0, c1, c2, sqrt_disc


    M = _M(D, opening_angle)

    W = diff(P, V)

    c0 = dot(np.dot(W, M), W)
    c1 = 2. * dot(np.dot(U, M), W)
    c2 = dot(np.dot(U, M), U)

    return _second_order_solutions(c2, c1, c0)

cdef double[:] cylinder_intersections(double[:] D,
                             double[:] P1,
                             double[:] P,
                             double[:] U,
                             double R2):

    cdef:

        double[:] W, A, B
        double c0, c1, c2, dot_UD, dot_WD

    W = diff(P, P1)

    dot_UD = dot(U, D)
    dot_WD = dot(W, D)

    A = diff(U, const_mult(D, dot_UD))
    B = diff(W, const_mult(D, dot_WD))

    c2 = dot(A, A)

    c1 = 2. * dot(A, B)

    c0 = dot(B, B) - R2

    return _second_order_solutions(c2, c1, c0)

cpdef _resolve_segment_direction(double[:] start_pos,
                                double[:] end_pos,
                                double start_rad, double end_rad):

        cdef:

            bint is_reversed = False
            double[:] P1, P2
            double r, R

        if start_rad - end_rad < EPS:

            P1 = start_pos
            P2 = end_pos
            r = start_rad
            R = end_rad

        if start_rad - end_rad > EPS:

            P1 = end_pos
            P2 = start_pos
            r = end_rad
            R = start_rad

            is_reversed = True

        return P1, P2, r, R, is_reversed


cpdef surface_intersect(const double[:, :] somata_positions,
                        const double[:, :] target_positions,
                        const double[:, :] segments_start,
                        const double[:, :] segments_end,
                        const double[:] sg_radii_start,
                        const double[:] sg_radii_end,
                        const unsigned long[:] somata_idx,
                        const unsigned long[:] target_idx,
                        const unsigned long[:] segment_idx,
                        const unsigned long[:, :] edges,
                        object graph):
    """ Calculates the intersection of segments that link target_positions along  the
    edges of the vasculature with somata positions.
    """

    adj = graph.adjacency_matrix

    cdef:

        unsigned long N = target_idx.size

        double[:, :] surface_target_positions = np.empty((N, 3), dtype=np.float64)
        #unsigned long[:] surface_target_idx = np.empty(N, dtype=np.uintp)
        unsigned long[:] surface_astros_idx = np.empty(N, dtype=np.uintp)
        unsigned long[:] vasculature_edge_idx = np.empty(N, dtype=np.uintp)

        # pos of tip of cone if not cylinder
        double[:] V  = np.empty(3, dtype=np.float64)

        # two points on the quadric (start, end)
        double[:] P1 = np.empty(3, dtype=np.float64)
        double[:] P2 = np.empty(3, dtype=np.float64)

        # direction of quadric from P1, P2
        double[:] D  = np.empty(3, dtype=np.float64)

        # parametric line point P + tU
        double[:] P  = np.empty(3, dtype=np.float64)
        double[:] U  = np.empty(3, dtype=np.float64)

        # start end of the segment to intersect withe the surface
        double[:] T0 = np.empty(3, dtype=np.float64)
        double[:] T1 = np.empty(3, dtype=np.float64)

        unsigned long astro_id, targt_id, sid, pid, cid
        int[:] cids, pids

        double r, R, cone_angle, T

        bint is_reversed

        unsigned long i = 0, n = 0, n_trials

        long[:] idx

    for i in range(somata_idx.size):

        astro_id = somata_idx[i]
        targt_id = target_idx[i]

        T0 = target_positions[targt_id].copy()
        T1 = somata_positions[astro_id].copy()

        # first segment id is the one that contains the target point
        sid = segment_idx[targt_id]
        n_trials = 0
        while n_trials < 10:

            # we have to determine if the intersection is with the current
            # segment or with a neighboring one. Therefore the intersection
            # is set to not_resolved until the correct segment is found or
            # the astro-target segment is contained in a truncated cone

            # parametric segment P = T0 + (T1 - T0)t
            P = T0
            U = diff(T1, T0)

            P1 = segments_start[sid].copy()
            P2 = segments_end[sid].copy()

            r = sg_radii_start[sid]
            R = sg_radii_end[sid]

            l = norm(P1, P2)

            # cylinder
            if fabs(r - R) < EPS:
                #print "norm : ", l
                # unit length direction of the cone
                D = const_mult(diff(P2, P1),  1. / l)

                t = cylinder_intersections(D, P1, P, U, R ** 2)

            # truncated cone
            else:
                # make sure that the vector is always from the small radius to the big one
                P1, P2, r, R, is_reversed = _resolve_segment_direction(P1, P2, r, R)

                #print "norm : ", l
                # unit length direction of the cone
                D = const_mult(diff(P2, P1),  1. / l)
                #print "r    : ", r, "R: ", R
                #print "P1   : ", np.asarray(P1)
                #print "P2   : ", np.asarray(P2)

                #print "D    : ", np.asarray(D)

                cone_angle = opening_angle(r, R, l)

                # coordinates of the tip of the cone
                V = tip_of_truncated_cone(D, P1, truncated_length(r, R, l))

                # target - astro segment angle with the line of the segment
                # min and max angles determined from the size of the caps

                t = cone_intersections(D, V, P, U, cone_angle)

            # segment extent validity
            if t[0] > -T_EPS and t[0] < 1. + T_EPS:

                T = t[0]

            elif t[1] > -T_EPS and t[1] < 1. + T_EPS:

                T = t[1]

            else:
                break

            P = parametric_point(P, U, T)

            left  = dot(D, diff(P, P1))
            right = dot(D, diff(P, P2))

            # after determining the point on the surface
            # validate its inclusion in the finite geometry
            if left > 0. and right < 0.:

               surface_target_positions[n] = P
               #surface_target_idx[n] = n
               surface_astros_idx[n] = astro_id
               vasculature_edge_idx[n] = sid

               n = n + 1
               break

            # check previous segment
            elif left < 0. and right < 0.:

                cid = edges[sid][0]
                pids = adj.parents(cid)

                # if there is no parent
                if pids.size > 0:
                    sid = graph.get_edge_index(pids[0], cid)
                else:
                    break

            # check next segment
            elif left > 0. and right > 0.:

                pid = edges[sid][1]
                cids = adj.children(pid)

                # if there are no childre
                if cids.size > 0:
                    sid = graph.get_edge_index(pid, cids[0])
                else:
                    break
            else:
                break

            n_trials = n_trials + 1

    return np.asarray(surface_target_positions[:n]),\
           np.asarray(surface_astros_idx[:n]),\
           np.asarray(vasculature_edge_idx[:n])


