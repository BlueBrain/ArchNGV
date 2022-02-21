import numpy as np
import pytest

from archngv.building.cell_placement import potentials


def test_lenard_jones():

    epsilon = 0.1
    r_m = 2.0

    distances = np.linspace(1.8, 2.5, 10)
    measured_potentials = potentials.lenard_jones(distances, r_m, epsilon)

    expected_potentials = np.array(
        [
            -0.02226467,
            -0.07885177,
            -0.09791630,
            -0.09910855,
            -0.09232481,
            -0.08251804,
            -0.07211150,
            -0.06222980,
            -0.05333483,
            -0.04555685,
        ]
    )

    assert np.allclose(measured_potentials, expected_potentials)


def test_coulomb():

    r_m = 2.0

    distances = np.linspace(1.8, 2.5, 10)

    measured_potentials = potentials.coulomb(distances, r_m)
    expected_potentials = np.array(
        [
            0.61728395,
            0.56720703,
            0.52298554,
            0.48374093,
            0.44875346,
            0.41742895,
            0.38927336,
            0.36387323,
            0.34088040,
            0.32,
        ]
    )

    assert np.allclose(measured_potentials, expected_potentials)


def test_inverse_distance():

    r_m = 2.0

    distances = np.linspace(1.8, 2.5, 10)

    measured_potentials = potentials.inverse_distance(distances, r_m)
    expected_potentials = np.array(
        [
            1.11111111,
            1.06508876,
            1.02272727,
            0.98360656,
            0.94736842,
            0.91370558,
            0.88235294,
            0.85308057,
            0.82568807,
            0.8,
        ]
    )

    assert np.allclose(measured_potentials, expected_potentials)


def test_spring():

    d_0 = 2.0

    spring_constant = 0.1

    distances = np.linspace(1.0, 3.0, 10)

    measured_potentials = potentials.spring(distances, d_0, spring_constant)
    expected_potentials = np.array(
        [
            0.10000000,
            0.06049383,
            0.03086420,
            0.01111111,
            0.00123457,
            0.00123457,
            0.01111111,
            0.03086420,
            0.06049383,
            0.1,
        ]
    )

    assert np.allclose(measured_potentials, expected_potentials)
