""" Extract an accurate representation of the geometry of the neuronal somata """

import os
import logging

import h5py
import numpy as np

from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist


L = logging.getLogger(__name__)


def apply_rotation_to_points(points, rotation_matrix):
    """Apply rotation matrix to array of 3d points"""
    return np.einsum('ij,kj->ik', points, rotation_matrix)


def principal_directions(points, return_eigenvalues=False):
    """ Find the principal directions of the covariance ellipsoid generated
    from the data points
    """
    covariance = np.cov(points.T)

    eigs, eigv = np.linalg.eig(covariance)

    idx = np.argsort(eigs)[::-1]

    if return_eigenvalues:

        return eigs[idx], eigv[:, idx].T

    else:

        return eigv[:, idx].T


def stable_marriage(women_preferences, men_preferences):
    '''Matches N women to M men so that max(M, N)
    are coupled to their preferred choice that is available
    See https://en.wikipedia.org/wiki/Stable_marriage_problem
    '''
    free_women = list(range(len(women_preferences)))
    free_men = list(range(len(men_preferences)))

    couples = {woman: None for woman in free_women}

    while len(free_women) > 0:

        m = free_men.pop()
        woman_of_choice = men_preferences[m].pop()

        if woman_of_choice in free_women:

            couples[woman_of_choice] = m
            free_women.remove(woman_of_choice)

        else:

            engaged_man = couples[woman_of_choice]

            if women_preferences[woman_of_choice].index(m) > engaged_man:

                free_men.append(engaged_man)
                couples[woman_of_choice] = m

            else:

                free_men.append(m)

    return couples


def find_matching(point_array1, point_array2):
    """ Finds the best match between two arrays of points
    """
    smallest_group, biggest_group = sorted((point_array1, point_array2), key=len)

    distx = cdist(smallest_group, biggest_group)

    s_preference = [np.argsort(row)[::-1].tolist() for row in distx]
    b_preference = [np.argsort(col)[::-1].tolist() for col in distx.T]

    return stable_marriage(s_preference, b_preference)


def plane_point_sgn(dir1, dir2, point):
    """ Given a plane defined by vectors dir1 and dir2
    calculates the side of the point by returning the
    sign of the determinant
    """
    M = np.column_stack((dir1, dir2, point))

    return np.linalg.det(M) > 0.


def rodrigues_rotation_matrix(axis, angle):
    '''
    Generates transformation matrix from unit vector
    and rotation angle. The rotation is applied in the direction
    of the axis which is a unit vector following the right hand rule.
    Inputs :
        axis : unit vector of the direction of the rotation
        angle : angle of rotation in rads
    Returns : 3x3 Rotation matrix
    '''
    def _sin(x):
        '''sine with case for pi multiples'''
        return 0. if np.isclose(np.mod(x, np.pi), 0.) else np.sin(x)

    def skew_symmetric_matrix(v):
        return np.array(((0., - v[2], v[1]), (v[2], 0., -v[0]), (- v[1], v[0], 0.)))

    u = axis / np.linalg.norm(axis)

    sn = _sin(angle)
    cs = np.cos(angle)

    A = skew_symmetric_matrix(u)

    return np.identity(3) + sn * A + (1. - cs) * np.linalg.matrix_power(A, 2)


def separate_contour_points_in_two_groups(max_dir, min_dir, contour_points):
    """ Create a plane that goes through the direction of maximum variation
    and the direction of minimum variation and split the contour_points in two groups
    using that plane.
    """
    left_right_mask = np.fromiter((plane_point_sgn(min_dir, max_dir, p) for p in contour_points), dtype=np.bool)

    group1_mask = left_right_mask
    group2_mask = ~ group1_mask

    if group1_mask.sum() <= group2_mask.sum():

        return contour_points[group1_mask],\
               contour_points[group2_mask]

    else:

        return contour_points[group2_mask],\
               contour_points[group1_mask]


def match_groups(group1, group2):
    """ Find a matching between group1 and group2
    """
    edge_dict = find_matching(group1, group2)
    return np.asarray(list(edge_dict.items()), dtype=np.intp)


def apply_edge_matching(edges, small_group, big_group):
    """ Dispatching of two groups
    """
    return small_group[edges[:, 0]], big_group[edges[:, 1]]


def generate_soma_points(contour_points):
    """ Better ask me for this part, it's a little bit complicated :)
    """
    max_dir, _, min_dir = principal_directions(contour_points)

    small_group, big_group = separate_contour_points_in_two_groups(max_dir, min_dir, contour_points)

    R = rodrigues_rotation_matrix(max_dir, np.pi)

    rotated_points = apply_rotation_to_points(small_group, R)

    edges = match_groups(rotated_points, big_group)

    small_group, big_group = apply_edge_matching(edges, small_group, big_group)

    vectors = big_group - small_group

    angles = np.linspace(0., 2. * np.pi, 20)

    offsets = small_group + 0.5 * vectors

    centered_points = small_group - offsets

    soma_points = []

    dirs = [np.cross(min_dir, vec / np.linalg.norm(vec)) for vec in vectors]
    dirs = [d / np.linalg.norm(d) for d in dirs]

    for angle in angles:

        Rs = [rodrigues_rotation_matrix(d, angle) for d in dirs]

        new_points = [np.dot(R, p) for R, p in zip(Rs, centered_points)]

        soma_points.extend(new_points + offsets)

    return np.asarray(soma_points, dtype=np.float)


def estimate_convex_hull_volume(points):
    """ From given points calculate convex hull volume
    """
    def vectorized_dot_product(v1s, v2s):
        """ Dot product applied to two arrays of points row by row
        """
        return np.einsum('ij,ij->i', v1s, v2s)

    def vectorized_triple_product(v1s, v2s, v3s):
        """ Gives the volume of a parallelepiped (vectorized for arrays of points)
        v1s . (v2s x v3s)
        """
        cross_product = np.cross(v2s, v3s)  # by default rowwise operation
        return vectorized_dot_product(v1s, cross_product)

    def volume_of_simplices(p1s, p2s, p3s, p4s):
        """ Gives the volume of a pyramid - tetrahedron aka 3-simplex
        """
        v1s = p3s - p1s
        v2s = p2s - p1s
        v3s = p4s - p1s

        return np.abs(vectorized_triple_product(v1s, v2s, v3s)) / np.math.factorial(3)

    def create_triangulation(points):
        """ Splits up the volume in tetrahedrals
        """
        dl = Delaunay(points)

        pyramids_idx = dl.simplices

        return (points[pyramids_idx[:, 0]],
                points[pyramids_idx[:, 1]],
                points[pyramids_idx[:, 2]],
                points[pyramids_idx[:, 3]])

    points = points[ConvexHull(points).vertices]

    p1s, p2s, p3s, p4s = create_triangulation(points)

    volumes = volume_of_simplices(p1s, p2s, p3s, p4s)

    return volumes.sum()


def get_soma_data(filename):
    """ Soma info from h5 file
    """
    with h5py.File(filename, 'r') as f:

        contour_points = f['points'][:f['structure'][1][0], :3]

        trunk_mask = f['structure'][:, 2] == 0

        section_offsets = f['structure'][trunk_mask, 0]

        trunk_starts = f['points'][section_offsets, :3]
        trunk_types = f['structure'][trunk_mask, 1]

    return contour_points, trunk_starts, trunk_types


def sphere_volume(radius):
    """ Volume of Sphere """
    return 4. / 3. * np.pi * radius ** 3


def filter_distant_neurite_starts(contour_points, trunk_starts):
    """ Removes trunk starts the distance of which to the center is bigger that two times
    the nax radius of the contour
    """
    centroid = np.mean(contour_points, axis=0)
    max_radius = np.max(np.linalg.norm(contour_points - centroid, axis=1))

    mask = np.linalg.norm(trunk_starts - centroid, axis=1) <= 2. * max_radius
    return trunk_starts[mask]


def volume_stats(filename):
    """ Get the contour points from the morphology files and calculate their
    distance to the center (0,0,0) of the morphology as radii. Their mean, max, min are
    calculated and stored. The volume is also approximated with a modified revolution of
    the contour points and their convex hull estimation.
    NOTE: if the morphology is not at the origin the center must be taken into account
    """
    contour_points, trunk_starts, _ = get_soma_data(filename)

    trunk_starts = filter_distant_neurite_starts(contour_points, trunk_starts)

    rot_soma_points = np.vstack((generate_soma_points(contour_points), trunk_starts))
    sim_soma_points = np.vstack((contour_points, trunk_starts))

    radii = np.linalg.norm(contour_points, axis=1)

    sim_cv_volume = estimate_convex_hull_volume(sim_soma_points)
    rot_cv_volume = estimate_convex_hull_volume(rot_soma_points)

    return np.mean(radii), np.max(radii), np.min(radii), rot_cv_volume, sim_cv_volume, rot_soma_points


def convex_envelope(points):
    """ Returns convex envelope of points """
    cv = ConvexHull(points)
    return cv.points[cv.vertices]


def _output_path(output_directory, gid):
    """ Output path """
    return os.path.join(output_directory, 'soma_' + str(gid) + '.h5')


def _worker(output_directory, gid, path):
    """ Write data worker """
    mean_radius, max_radius, min_radius, rot_vol, sim_vol, soma_points = volume_stats(path)

    output_path = _output_path(output_directory, gid)

    with h5py.File(output_path, 'w') as fd:

        fd.create_dataset('points', data=soma_points, dtype=np.float)
        fd.attrs['mean_radius'] = mean_radius
        fd.attrs['max_radius'] = max_radius
        fd.attrs['min_radius'] = min_radius

        fd.attrs['volume_contour_trunks'] = sim_vol
        fd.attrs['volume_revolution_solid'] = rot_vol


def extract_neuronal_somata_information(output_directory, neuronal_microcircuit, map_func=map):
    """ summary function """
    neuronal_gids = neuronal_microcircuit.v2.cells.ids()

    L.info('Getting filepaths...')

    paths = list(map(neuronal_microcircuit.v2.morph.get_filepath, neuronal_gids))

    n_cells = len(paths)

    data = np.zeros((len(paths), 5), dtype=np.float)

    data = {'max_radii': np.zeros(n_cells, dtype=np.float),
            'min_radii': np.zeros(n_cells, dtype=np.float),
            'mean_radii': np.zeros(n_cells, dtype=np.float),
            'volume_contour_trunks': np.zeros(n_cells, dtype=np.float),
            'volume_revolution_solid': np.zeros(n_cells, dtype=np.float)}

    L.info('calculating volumes')

    outputs = list(map_func(volume_stats, paths))

    somata_geom_path = os.path.join(output_directory, 'microcircuit_somata_geometries_by_revolution.h5')

    L.info('Writing to h5 file.')
    with h5py.File(somata_geom_path, 'w') as f:
        for n in range(len(neuronal_gids)):

            output = outputs[n]

            data['mean_radii'][n] = output[0]
            data['max_radii'][n] = output[1]
            data['min_radii'][n] = output[2]
            data['volume_contour_trunks'] = output[3]
            data['volume_revolution_solid'] = output[4]
            points = output[5]

            f.create_dataset(str(n), data=convex_envelope(points))

    positions = neuronal_microcircuit.v2.cells.positions().values[:len(neuronal_gids)]

    pdata_path = os.path.join(output_directory, 'microcircuit_point_data_min_radii.npy')
    np.save(pdata_path, np.column_stack((positions, data['min_radii'])))

    pdata_path = os.path.join(output_directory, 'microcircuit_point_data_max_radii.npy')
    np.save(pdata_path, np.column_stack((positions, data['max_radii'])))

    pdata_path = os.path.join(output_directory, 'microcircuit_point_data_mean_radii.npy')
    np.save(pdata_path, np.column_stack((positions, data['mean_radii'])))

    radii_path = os.path.join(output_directory, 'microcircuit_neuronal_radii_estimations.npy')

    np.save(radii_path, np.column_stack((data['mean_radii'],
                                         data['max_radii'],
                                         data['min_radii'])))

    # with open('morph_soma_points.pkl', 'w') as fp:
    #    pickle.dump(outputs, fp)
