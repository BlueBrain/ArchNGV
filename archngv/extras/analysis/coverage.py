import numpy as np

def is_inside_union_of_spheres(point, points, radii):
    return np.any(np.linalg.norm(points - point, axis=1) <= radii)


def monte_carlo_volume_estimation(function, xmin, xmax, ymin, ymax, zmin, zmax, epsi, N_max):

    V = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    N_included = 0
    N_total = 0

    while N_total < N_max:

        N_total += 1

        point = np.random.uniform((xmin, ymin, zmin), (xmax, ymax, zmax))

        N_included += int(function(point))

    return V * float(N_included) / float(N_total)


def somata_coverage_using_spheres(positions, radii, bounding_box):

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box.ranges.T

