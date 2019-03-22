import numpy as np
from scipy.spatial import cKDTree


def mask_points_inside_the_circuit_bounding_box(points, tile_vertices, circuit):

    # bounding box values of mesocircuit
    meso_xmin, meso_xmax = tile_vertices[..., 0].min(), tile_vertices[..., 0].max()
    meso_zmin, meso_zmax = tile_vertices[..., 1].min(), tile_vertices[..., 1].max()
    meso_ymin, meso_ymax = circuit._ymin, circuit._ymax

    # boolean mask for points inside the bounding box
    return (meso_xmin <= points[:, 0]) & (points[:, 0] <= meso_xmax) & \
           (meso_ymin <= points[:, 1]) & (points[:, 1] <= meso_ymax) & \
           (meso_zmin <= points[:, 2]) & (points[:, 2] <= meso_zmax)

def mask_points_inside_tiles(points, closest_tile_vertices):

    # calculate the projection of the inclined side
    h = np.cos(np.pi / 3.) * np.linalg.norm(closest_tile_vertices[0, 1, :] - closest_tile_vertices[0, 0, :])

    # tile coordinate extrema
    ctv_xmin = closest_tile_vertices[..., 0].min(axis=1)
    ctv_xmax = closest_tile_vertices[..., 0].max(axis=1)

    ctv_zmin = closest_tile_vertices[..., 1].min(axis=1)
    ctv_zmax = closest_tile_vertices[..., 1].max(axis=1)

    # the points that reside inside the square included in the hexagon
    return (ctv_xmin + h <= points[:, 0]) & (points[:, 0] <= ctv_xmax - h) & \
           (ctv_zmin     <= points[:, 1]) & (points[: ,1] <= ctv_zmax)

def mask_points_inside_triangle(xz_points, rem_tile_vertices):

    """
        A  C  
       /|  |\
      / |  | \
     B\ |  | /B
       \|  |/
        C  A
    """

    h = np.cos(np.pi / 3.) * np.linalg.norm(rem_tile_vertices[0, 1, :] - rem_tile_vertices[0, 0, :])

    # shift will be 0 or 1 corresponding to the left and right triangle respectively
    shift = (xz_points[:, 0] > rem_tile_vertices[..., 0].max(axis=1) - h).astype(np.int)

    # choose the triangle by shifting the indices of the vertices by 3 if the right triangle is chosen
    adv_index_dim_1 = np.ones(rem_tile_vertices.shape[0], dtype=np.bool)

    # two sets of indices: (-1, 0, 1) or (2, 3, 4) for left and right triangles of the hex
    index_A = 3 * shift - 1
    index_B = index_A + 1
    index_C = index_B + 1

    # triangle points
    A = rem_tile_vertices[adv_index_dim_1, index_A, :]
    B = rem_tile_vertices[adv_index_dim_1, index_B, :]
    C = rem_tile_vertices[adv_index_dim_1, index_C, :]

    # vectors
    AB = B - A
    AC = C - A
    AP = xz_points - A

    # 2d cross products on the k axis
    cross_AC_AP = AC[:, 0] * AP[:, 1] - AC[:, 1] * AP[:, 0]
    cross_AB_AP = AB[:, 0] * AP[:, 1] - AB[:, 1] * AP[:, 0]
    cross_AC_AB = AC[:, 0] * AB[:, 1] - AC[:, 1] * AB[:, 0]

    t = cross_AC_AP / cross_AC_AB
    r = - cross_AB_AP / cross_AC_AB

    return (t >= 0) & (r >= 0) & (np.abs(r) + np.abs(t) <= 1.)


def mask_points_by_geometry(points, circuit):
    """
      zmax    zmax
      xmin    xmax
        _______
       /       \
      /         \
      \         /
       \_______/

      zmin     zmin
      xmin     xmax

    """
    tile_centers = circuit.centers
    tile_vertices = circuit.vertices

    # points inside the bounding box of the circuit
    mask_inside = mask_points_inside_the_circuit_bounding_box(points, tile_vertices, circuit)

    # if there is no point inside the geometry return an empty array
    if not mask_inside.any():
        return np.array([])

    # remove points that are completely outside
    points = points[mask_inside, :]

    # KDTree to find the closest tile center for each point on the xz projection
    t = cKDTree(tile_centers)
    distances, nn_indices = t.query(points[:, (0, 2)])

    closest_tile_vertices = tile_vertices[nn_indices, ...]

    # mask of points that reside in the square included in the hexagonal tiles
    mask_inside = mask_points_inside_tiles(points[:, (0, 2)], closest_tile_vertices)

    if mask_inside.all():
        return points[mask_inside, :]

    # there is no need to apply further checks to the points that are inside the geometry
    # we have to check the shady ones near the triangular sides of the hexagons
    rem_points = points[~mask_inside, :]

    rem_tile_vertices = closest_tile_vertices[~mask_inside, ...]

    # check if remaning points are in the triangular sides of the hexagonal tiles
    mask_rem_inside = mask_points_inside_triangle(rem_points[:, (0, 2)], rem_tile_vertices)

    #print points.shape, rem_points.shape, mask_inside.sum(), mask_rem_inside.sum()
    return np.vstack((points[mask_inside, :], rem_points[mask_rem_inside, :]))


def mask_indices_by_geometry(points, circuit):
    """
      zmax    zmax
      xmin    xmax
        _______
       /       \
      /         \
      \         /
       \_______/

      zmin     zmin
      xmin     xmax

    """
    tile_centers = circuit.centers
    tile_vertices = circuit.vertices

    # points inside the bounding box of the circuit
    B = mask_points_inside_the_circuit_bounding_box(points, tile_vertices, circuit)

    # if there is no point inside the geometry return an empty array
    if not B.any(): return np.array([])

    # KDTree to find the closest tile center for each point on the xz projection
    t = cKDTree(tile_centers)
    distances, nn_indices = t.query(points[:, (0, 2)])

    closest_tile_vertices = tile_vertices[nn_indices]

    # mask of points that reside in the square included in the hexagonal tiles
    I1 = mask_points_inside_tiles(points[:, (0, 2)], closest_tile_vertices)

    if I1.sum() == B.sum(): return I1.nonzero()[0]

    # check if remaning points are in the triangular sides of the hexagonal tiles
    I2 = mask_points_inside_triangle(points[:, (0, 2)], closest_tile_vertices)

    I = B & (I1 | I2)
    #print points.shape, rem_points.shape, mask_inside.sum(), mask_rem_inside.sum()
    return I.nonzero()[0]
