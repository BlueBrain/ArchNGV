from collections import deque
import multiprocessing
import numpy as np

from archngv import MicrodomainTesselation
from archngv.spatial.collision import convex_shape_with_spheres
from archngv.spatial.bounding_box import BoundingBox

N_POINTS = 1000000

"""
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
"""

def unpack_arguments(func):
    def wrapped(tuple_arguments):
        return func(*tuple_arguments)
    return wrapped


def calculate_overlap(microdomains, index, points):

    are_overlapping = np.ones(N_POINTS, dtype=np.bool)

    visited = set([index])
    q = deque([index])

    while q:

        domain = microdomains[q.pop(0)]

        are_inside = convex_shape_with_spheres(domain.face_points, domain.face_normals, points)

        if not are_inside.any():
            continue

        are_overlapping &= are_inside

        for neighbor_id in domain.neighbor_ids:
            if neighbor_id >= 0 and neighbor_id not in visited:
                q.append(neighbor_id)
                visited.add(neighbor_id)

    return are_overlapping.sum()


def create_sample_from_bbox(bbox):

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox.ranges
    return np.random.uniform((xmin, ymin, zmin), (xmax, ymax, zmax), size=N_POINTS)



@unpack_arguments
def microdomain_worker(tesselation_filepath, index):

    microdomains = MicrodomainTesselation(tesselation_filepath)

    # is boundary domain
    if (microdomains.domain_neighbors(index) == -1).any():
        return True, None, None

    domain = microdomains[index]

    bbox = BoundingBox.from_points(domain)
    points = create_sample_from_bbox(bbox, N_POINTS)

    # number of points overlapping
    n_overlapping = calculate_overlap(microdomains, index, points)

    # monte carlo volume
    overlap_volume = bbox.volume * float(n_overlapping) / float(N_POINTS)

    return False, overlap_volume, domain.volume


def microdomain_tesselation_overlap_distribution(filepath):

    n_cpus = multiprocessing.cpu_count()
    n_microdomains = len(MicrodomainTesselation(filepath))

    with multiprocessing.Pool(n_cpus) as pool:

        inputs = ((filepath, index) for index in range(n_microdomains))
        res_gen = pool.imap_unordered(microdomain_worker, inputs)

        results = [(overlap_vol, vol) for is_boundary, overlap_vol, vol in res_gen if not is_boundary]
        results = np.asarray(results, dtype=np.float)

        print(results)
