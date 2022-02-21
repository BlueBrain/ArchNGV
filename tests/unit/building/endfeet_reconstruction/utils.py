import numpy as np


def create_mesh_data(height=10, width=10, world_offset=0):
    """
    Build a triangle strip:
    Vertices are layed out left to right, bottom to top:

    8--9-10-11
    |/ |/ |/ |
    4--5--6--7
    |/ |/ |/ |
    0--1--2--3

    Triangles are built from the vertices clockwise (0, 5, 1), (0, 4, 5)
    """
    np.random.seed(0)
    lower_triangles = [(i, i + width + 1, i + 1) for i in range(width - 1)]
    upper_triangles = [(i, i + width, i + width + 1) for i in range(width - 1)]

    triangles = np.vstack((lower_triangles, upper_triangles))
    mesh_triangles = (
        np.repeat(np.arange(height - 1), len(triangles))[:, None] * [width, width, width]
        + np.tile(triangles, (height - 1, 1))
    ).astype(np.uint64)
    mesh_triangles += world_offset

    mesh_coordinates = np.transpose(
        [np.tile(np.arange(width), height), np.repeat(np.arange(height), width)]
    )
    mesh_coordinates = np.vstack((mesh_coordinates.T, [1] * len(mesh_coordinates))).T
    mesh_coordinates = np.vstack((np.zeros((world_offset, 3)), mesh_coordinates)).astype(np.float32)
    return mesh_coordinates, mesh_triangles, np.unique(mesh_triangles)
