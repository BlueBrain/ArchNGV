import numpy as np

import archngv.utils.generics as tested


def test_is_iterable():
    testee = {1, 2, 3}
    assert tested.is_iterable(testee)

    testee = {1: 1, 2: 2, 3: 3}
    assert tested.is_iterable(testee)

    testee = [1, 2, 3]
    assert tested.is_iterable(testee)

    testee = np.asarray([1, 2, 3])
    assert tested.is_iterable(testee)

    testee = "aaaaa"
    assert not tested.is_iterable(testee)


def test_ensure_list():
    expected = [1, 2, 3]

    testee = {1, 2, 3}
    assert tested.ensure_list(testee) == expected

    testee = {1: 1, 2: 2, 3: 3}
    assert tested.ensure_list(testee) == expected

    testee = [1, 2, 3]
    assert tested.ensure_list(testee) == expected

    testee = np.asarray([1, 2, 3])
    assert tested.ensure_list(testee) == expected

    testee = "aaaaa"
    assert tested.ensure_list(testee) == [testee]
