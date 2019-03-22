import numpy as np


def _spherical_shell_volume(rinner, router):
    #return (4. / 3.) * np.pi * (router ** 3 - rinner ** 3) 
    r = router
    dr = r - rinner
    return 2. * np.pi * dr * r ** 2


def _find_interior_points(x, y, z, S, rMax):

    return np.where( (x > rMax) & (x < S - rMax) & \
                     (y > rMax) & (y < S - rMax) & \
                     (z > rMax) & (z < S - rMax) )[0]


def _total_particle_density(n_particles, box_side):
    return n_particles / box_side ** 3

def pairCorrelationFunction_3D(x, y, z, S, rMax, dr):
    """Compute the three-dimensional pair correlation function for a set of
    spherical particles contained in a cube with side length S.  This simple
    function finds reference particles such that a sphere of radius rMax drawn
    around the particle will fit entirely within the cube, eliminating the need
    to compensate for edge effects.  If no such particles exist, an error is
    returned.  Try a smaller rMax...or write some code to handle edge effects! ;)
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        z               an array of z positions of centers of particles
        S               length of each side of the cube in space
        rMax            outer diameter of largest spherical shell
        dr              increment for increasing radius of spherical shell
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        spherical shells used to compute g(r)
        reference_indices   indices of reference particles
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram

    # Find particles which are close enough to the cube center that a sphere of radius
    # rMax will not cross any face of the cube

    interior_indices = _find_interior_points(x, y, z, S, rMax)
    num_interior_particles = len(interior_indices)

    assert num_interior_particles > 1

    spherical_shell_radii = np.arange(0., rMax + 1.1 * dr, dr)

    g = np.zeros([num_interior_particles, len(spherical_shell_radii) - 1])

    # Compute pairwise correlation for each interior particle
    for i, index in enumerate(interior_indices):

        d = np.sqrt((x[index] - x) ** 2 + (y[index] - y) ** 2 + (z[index] - z) ** 2)

        # make sure that the distance to self will not be accounted
        d[index] = 2. * rMax

        g[i, :] = np.histogram(d, bins=spherical_shell_radii, normed=False)[0]

    # Average g(r) for all interior particles and compute radii

    # total particle density
    rho = _total_particle_density(len(x), S)

    radii = spherical_shell_radii[1:]
    g_average = g.mean(axis=0) / (rho *  (4./3.) * np.pi * (radii ** 3 - (radii - dr) ** 3))

    return (g_average, radii, interior_indices)


def RipleyKFunction(x, y, z, S, rMax, dr):

    interior_indices = _find_interior_points(x, y, z, S, rMax)

    num_interior_particles = len(interior_indices)

    assert num_interior_particles > 1

    spherical_shell_radii = np.arange(0., rMax + 1.1 * dr, dr)

    k = np.zeros((num_interior_particles, len(spherical_shell_radii)), dtype=np.float)

    for i, index in enumerate(interior_indices):

        d2 = np.sqrt((x[index] - x) ** 2 + (y[index] - y) ** 2 + (z[index] - z) ** 2)

        # make sure that the distance to self will not be accounted
        d2[index] = 4. * rMax ** 2

        #mask = np.ones_like(d2, dtype=np.bool)

        for j, radius in enumerate(spherical_shell_radii):
            k[i, j] = (d2 <= radius).sum()


    # total particle density
    rho = _total_particle_density(len(x), S)

    k_average = k.mean(axis=0) / rho

    return (k_average, spherical_shell_radii, interior_indices)


def RipleyLFunction(x, y, z, S, rMax, dr):
    (k_average, spherical_shell_radii, interior_indices) = RipleyKFunction(x, y, z, S, rMax, dr)
    return np.sqrt(k_average / np.pi), spherical_shell_radii, interior_indices
