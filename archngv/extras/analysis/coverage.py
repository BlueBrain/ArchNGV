from collections import deque
import logging
import multiprocessing
import numpy as np
import pandas as pd

from archngv import NGVConfig, NGVCircuit
from archngv.spatial.collision import convex_shape_with_spheres
from archngv.spatial.bounding_box import BoundingBox

from .common import find_layer

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


N_POINTS = 100000


def _spherical_extent(domain):
    """ Returns the extent of the domain as two times
    the distance its furthest point.
    """
    vectors = domain.points - domain.centroid
    lengths = np.linalg.norm(vectors, axis=1)
    return 2. * lengths.max()


def _overlap_fraction(microdomains, domain_points, domain_neighbors, index, scale_factor):
    """
    Find the number of points that are shared by the microdomain with index
    and any other domain in the neighborhood.

    Args:
        microdomains: MicrodomainTesselation
            The domain tesselation interface.
        domain_points:
            The sample of uniform points that lie inside the current domain.
        index: int
            The id of the current domain.

    Returns:
        fraction: float
            The fraction of sample points in the domain that are shared with
            neighbors N_shared / N_domain_points
    """
    # the overlap points belong to both the domain and a neighbor
    overlap_mask = np.zeros(len(domain_points), dtype=np.bool)

    visited = set([index])
    q = deque()

    for neighbor_id in domain_neighbors:
        if neighbor_id >= 0 and neighbor_id not in visited:
            q.append(neighbor_id)
            visited.add(neighbor_id)

    n_neighbors = 0

    while q:

        # get domain in neighborhood and scale it wrt to the scale factor
        neighbor = microdomains[q.popleft()].scale(scale_factor)

        # find points that are inside the neighbor
        inside_neighbor = convex_shape_with_spheres(neighbor.face_points, neighbor.face_normals, domain_points)

        if not inside_neighbor.any():
            continue

        n_neighbors += 1

        # the overlap are the domain points that are shared with any neighbor (or)
        overlap_mask |= inside_neighbor

        # add the neighbors of this domain that are not boundary or visited
        for neighbor_id in neighbor.neighbor_ids:
            if neighbor_id >= 0 and neighbor_id not in visited:
                q.append(neighbor_id)
                visited.add(neighbor_id)

    # monte carlo volume
    return float(overlap_mask.sum()) / float(len(overlap_mask)), n_neighbors


def _distance_to_vasculature(circuit, index):
    """ Returns average distance of the current microdomain to the vasculature
    surface points.
    """
    soma_center = circuit.data.astrocytes.astrocyte_positions[index]

    gv_conn = circuit.connectome.gliovascular
    gv_data = circuit.data.endfeetome.targets

    ids = gv_conn.astrocyte.to_endfoot(index)

    if len(ids) == 0:
        return soma_center, 0.0

    points = gv_data.endfoot_surface_coordinates[ids]
    return soma_center, np.linalg.norm(points - soma_center, axis=1).mean()


def _points_in_domain(domain):

    # get a point sample in the bbox of the domain
    bbox = BoundingBox.from_points(domain.points)
    points = np.random.uniform(bbox.min_point, bbox.max_point, size=(N_POINTS, 3))

    # reduce the point cloud to the inside of the domain
    inside_domain = convex_shape_with_spheres(domain.face_points, domain.face_normals, points)
    return points[inside_domain]


def microdomain_worker(tup):
    """
    Microdomain worker that extracts information concerning a microdomain.
    """
    config_path, index, scale_factor = tup

    circuit = NGVCircuit(NGVConfig.from_file(config_path))
    microdomains = circuit.data.microdomains
    # get the domains and scale wrt to the scale factor
    domain = microdomains[index].scale(scale_factor)
    domain_points = _points_in_domain(domain)

    # fraction of domain points that belong to the overlap
    fraction, n_neighbors = _overlap_fraction(microdomains, domain_points, domain.neighbor_ids, index, scale_factor)

    domain_volume = domain.volume
    overlap_volume = fraction * domain_volume

    spherical_extent = _spherical_extent(domain)
    soma_center, dist_to_vasculature = _distance_to_vasculature(circuit, index)


    bbox = circuit.data.vasculature.bounding_box

    layer = find_layer(soma_center[1])

    print(index, soma_center, layer, scale_factor, spherical_extent, overlap_volume / domain_volume)

    return (
        layer,
        scale_factor,
        spherical_extent,
        domain_volume,
        overlap_volume,
        overlap_volume / domain_volume,
        dist_to_vasculature,
        n_neighbors)


def _create_ids(config_path, n_samples):
    """ It returns a random selection of n_samples microdomain ids
    that are not boundaries.
    """
    microdomains = NGVCircuit(NGVConfig.from_file(config_path)).data.microdomains
    n_microdomains = len(microdomains)

    L.info('Microdomain sample size: %d', n_samples)

    shuffled_ids = np.random.choice(np.arange(n_microdomains), n_microdomains)

    ids = np.empty(n_samples, dtype=np.int)

    n = 0
    while n < n_samples:
        index = shuffled_ids[n]
        if not (microdomains.domain_neighbors(index) == -1).any():
            ids[n] = index
            n += 1

    return ids


def microdomain_tesselation_measurements(circuit_config_path):

    np.random.seed(0)

    n_cpus = multiprocessing.cpu_count()
    n_samples = 20 * n_cpus

    ids = _create_ids(circuit_config_path, n_samples)

    L.info('ids: %s', ids)

    #scale_factors = [1.01, 1.05]
    scale_factors = np.arange(1.01, 1.5, 0.01)
    data = []

    with multiprocessing.Pool(n_cpus) as pool:

        map_func = pool.imap_unordered

        for i, scale_factor in enumerate(scale_factors):
            inputs = ((circuit_config_path, index, scale_factor) for index in ids)
            data.append(list(map_func(microdomain_worker, inputs)))

    data = np.vstack(data)

    labels = [
        'layer',
        'scale_factor',
        'spherical_extent',
        'domain_volume',
        'overlap_volume',
        'overlap_fraction',
        'avg_distance_to_endfeet',
        'overlapping_neighbors'
    ]

    dset = pd.DataFrame({label: data[:, i] for i, label in enumerate(labels)})

    dset.to_pickle('microdomain_overlap.pkl')
    np.save('microdomain_overlap.npy', data)
    return dset
