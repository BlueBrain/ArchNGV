from archngv.utils import functional as _funct


def test_consecutive_pairs():

    a = [5, 3, 1, 4, 0]

    gen = _funct.consecutive_pairs(a)

    expected_a = [(5, 3), (3, 1), (1, 4), (4, 0)]

    assert list(gen) == expected_a
