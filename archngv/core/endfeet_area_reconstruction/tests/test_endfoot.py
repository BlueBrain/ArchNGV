import utils
from archngv.core.endfeet_area_reconstruction.detail import endfoot


def create_data(height=10, width=10):
    ef = endfoot.create_endfoot_from_global_data(0,
                                                 *utils.create_mesh_data(height,
                                                                         width))
    return ef


def test_EndFoot():
    height = width = 10
    ef = create_data(height, width)
    #ef.coordinates_array
    #ef.triangle_array
    #ef.edges

    assert ef.local_to_global_map == {i: i for i in range(ef.number_of_vertices)}

    assert list(ef.vasculature_vertices) == list(range(ef.number_of_vertices))

    assert ef.number_of_vertices == height * width
    assert ef.number_of_triangles == (height - 1.) * (width - 1.) * 2.
    assert ef.area == (height - 1.) * (width - 1.)

    assert ef.vertex_neighbors[0] == {1, 10, 11}
    assert ef.vertex_neighbors[11] == {0, 1, 10, 12, 21, 22}

    assert ef.vertex_to_triangles[0] == {0, 9}
    assert ef.vertex_to_triangles[98] == {160, 161, 151}
    assert ef.vertex_to_triangles[55] == {104, 76, 85, 86, 94, 95}

    assert ef.edge_to_triangles[frozenset((94, 95))] == {157}
    assert ef.edge_to_triangles[frozenset((60, 61))] == {99, 108}


def test_EndFoot_shrink():
    #remove all
    ef = create_data()
    ef.shrink(set(range(ef.number_of_vertices)))
    assert ef.area == 0
    assert ef.number_of_vertices == 0
    assert ef.number_of_triangles == 0

    #remove left hand triangle strip
    height = width = 10
    ef = create_data(height, width)
    ef.shrink(set(range(0, height * width, width)))
    assert ef.area == (height - 1) * (width - 2)
    assert ef.number_of_vertices == height * (width - 1)
    assert ef.number_of_triangles == (height - 1) * (width - 2) * 2
