from ..pattern import SpatialSpherePattern


def test_constructor():

    max_spheres = 10

    pat = SpatialSpherePattern(max_spheres)

    assert len(pat._coordinates) == max_spheres
    assert len(pat._radii) == max_spheres
    assert pat._index == 0
    assert len(pat._si) == 0


def test_getitem():
    pass
