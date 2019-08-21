# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


from libc.math cimport fabs, sqrt, fmin


DEF EPS = 1e-6

cdef inline float diagonal_triple_product(float ax, float ay, float az,
                                           float Qxx, float Qyy, float Qzz,
                                           float bx, float by, float bz) nogil:
    return ax * Qxx * bx + ay * Qyy * by + az * Qzz * bz

cdef inline void  second_order_solutions(float a, float b, float c, float* root1, float* root2) nogil:
    # assumes that root1 and root2 are initialized as -1

    # numerical innacuracy
    if fabs(b) < EPS:
        b = 0.

    #  first order equation if a is zero
    if fabs(a) < EPS:
        if b != 0.:
            root1[0] = - c / b
        return

    cdef float q = b ** 2 - 4. * a * c

    #numerical inaciracy
    if fabs(q) < EPS:
        q = 0.

    # no real roots, nothing to do here
    if q < 0.:

        return

    elif q > 0.:

        # stable variants
        if b >= 0.:

            root1[0] = 0.5 * (- b - sqrt(q)) / a

            root2[0] = 2. * c / (- b - sqrt(q))

        else:

            root1[0] = 0.5 * (- b + sqrt(q)) / a

            root2[0] = 2. * c / (- b + sqrt(q))

    else:

        root1[0] = - c / (2. * a)



cdef float local_solver_2D(float Ax, float Ay, float Az,
                            float Bx, float By, float Bz,
                            float Cx, float Cy, float Cz,
                            float TA, float TB, float Qxx, float Qyy, float Qzz) nogil:

    """
    Update the travel time at C taking into account
    the upwind neighbors A and B.

           C (v3)
          / \
         /   \
        /     \
       /       \
      /         \
      - - - - - -
    A (v1)       B (v2)

    T1 is the travel time of the wavefront to A
    T2 is the travel time of the wavefront to B

    Q is the speed tensor, determining the spatial anisotropy of the speed of the wave. It
    can be also a field by assigning a different one to each vertex
    """
    # AC vector
    cdef float ACx = Cx - Ax
    cdef float ACy = Cy - Ay
    cdef float ACz = Cz - Az

    # AB vector
    cdef float ABx = Bx - Ax
    cdef float ABy = By - Ay
    cdef float ABz = Bz - Az

    # BC vector
    cdef float BCx = Cx - Bx
    cdef float BCy = Cy - By
    cdef float BCz = Cz - Bz

    # a = AC
    # b = AB
    # c = BC

    cdef float Caa = diagonal_triple_product(ACx, ACy, ACz, Qxx, Qyy, Qzz, ACx, ACy, ACz)
    cdef float Cab = diagonal_triple_product(ACx, ACy, ACz, Qxx, Qyy, Qzz, ABx, ABy, ABz)
    cdef float Cba = Cab
    cdef float Cbb = diagonal_triple_product(ABx, ABy, ABz, Qxx, Qyy, Qzz, ABx, ABy, ABz)

    cdef float TAB = TB - TA


    cdef float l1 = -1.0
    cdef float l2 = -1.0

    cdef float* l1_p = &l1
    cdef float* l2_p = &l2

    cdef float A, B, C

    cdef bint l1_valid, l2_valid

    cdef float triple1, triple2
    cdef float T31, T32

    if fabs(TAB) < EPS:

        root1 = 0.5 * (Cab + Cba) / Cbb

    else:

        inv_TAB_sq = 1. / TAB ** 2

        A = Cbb * (1. - Cbb * inv_TAB_sq)
        B = (Cab + Cba) * (-1. + Cbb * inv_TAB_sq)
        C = Caa - 0.25 * (Cab + Cba) ** 2 * inv_TAB_sq

        second_order_solutions(A, B, C, l1_p, l2_p)

    l1_valid = 0. <= l1 <= 1
    l2_valid = 0. <= l2 <= 1.

    # solutions can be symmetric. In that case pick the one that gives the shortest
    # travel time (Fermat's principle).
    if l1_valid and l2_valid:


        triple1 = diagonal_triple_product(ACx - l1 * ABx, ACy - l1 * ABy, ACz - l1 * ABz,
                                          Qxx, Qyy, Qzz,
                                          ACx - l1 * ABx, ACy - l1 * ABy, ACz - l1 * ABz)

        triple2 = diagonal_triple_product(ACx - l2 * ABx, ACy - l2 * ABy, ACz - l2 * ABz,
                                          Qxx, Qyy, Qzz,
                                          ACx - l2 * ABx, ACy - l2 * ABy, ACz - l2 * ABz)

        T31 = TA + l1 * TAB + sqrt(triple1)
        T32 = TA + l2 * TAB + sqrt(triple2)

        return fmin(T31, T32)

    elif l1_valid and not l2_valid:
        return TA + l1 * TAB + sqrt(diagonal_triple_product(ACx - l1 * ABx, ACy - l1 * ABy, ACz - l1 * ABz,
                                                   Qxx, Qyy, Qzz,
                                                   ACx - l1 * ABx, ACy - l1 * ABy, ACz - l1 * ABz))

    elif l2_valid and not l1_valid:
        return TA + l2 * TAB + sqrt(diagonal_triple_product(ACx - l2 * ABx, ACy - l2 * ABy, ACz - l2 * ABz,
                                                            Qxx, Qyy, Qzz,
                                                            ACx - l2 * ABx, ACy - l2 * ABy, ACz - l2 * ABz))

    # if no solution is found the characteristic of the gradient is outside the
    # triangle. In that case give the smallest travel time through the edges of
    # the triange
    else:

        T31 = TA + sqrt(Caa)
        T32 = TB + sqrt(diagonal_triple_product(BCx, BCy, BCz,
                                                Qxx, Qyy, Qzz,
                                                BCx, BCy, BCz))

        return fmin(T31, T32)

