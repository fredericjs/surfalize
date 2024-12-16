import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from surfalize import Surface

target = {
    'level':
    np.array([[ 0.05907043, -0.42588871,  0.38703552, -0.02417648, -0.11975427],
              [-0.28006027,  0.2412264 , -0.06524836,  0.05864385, -0.4793917 ],
              [ 0.10925432,  0.11532314,  0.13177001,  0.47019269,  0.21987351],
              [-0.11263856, -0.02350592,  0.24870192, -0.37709521,  0.24105463],
              [ 0.23472611, -0.2139206 , -0.28376827, -0.08565762, -0.0257666 ]]),

    'detrend_polynomial_deg=2':
    np.array([[0.15432879, -0.36708897, 0.43450201, 0.03708213, -0.01957816],
              [-0.29246027, 0.19175307, -0.12666966, 0.01039996, -0.48933282],
              [0.06178783, 0.03016860, 0.03405278, 0.38503815, 0.17240702],
              [-0.12257968, -0.07174981, 0.18728062, -0.42656854, 0.22865464],
              [0.33490222, -0.15266199, -0.23630178, -0.02685788, 0.06949175]]),

    'detrend_polynomial_deg=5':
    np.array([[0.05218090, -0.16516211, 0.22388561, -0.16100846, 0.05010407],
              [-0.12588806, 0.27631249, -0.23954774, 0.15371023, -0.06458693],
              [0.12961399, -0.09810559, -0.23445923, 0.24477929, -0.04182846],
              [-0.09028739, 0.02792215, 0.29201927, -0.30665543, 0.07700140],
              [0.03438056, -0.04096694, -0.04189790, 0.06917437, -0.02069009]])

}

@pytest.fixture
def data():
    np.random.seed(0)
    initial_data = np.random.uniform(size=(5, 5))
    return initial_data

@pytest.fixture
def noisy_surface(surface):
    np.random.seed(0)
    mask = (np.random.normal(size=surface.size) > 4).astype('int')
    noise_surface = surface + Surface(mask * 10, surface.step_x, surface.step_y)
    return noise_surface

def test_level(data):
    y, x = np.indices(data.shape)
    output = Surface(np.random.uniform(size=(5, 5)) + x * 5, 1, 1).level().data
    assert_array_almost_equal(output, target['level'])

def test_detrend_polynomial(data):
    y, x = np.indices(data.shape)
    output = Surface(np.random.uniform(size=(5, 5)) + x ** 2, 1, 1).detrend_polynomial(degree=2).data
    assert_array_almost_equal(output, target['detrend_polynomial_deg=2'])
    # output = Surface(np.random.uniform(size=(5, 5)) + x ** 5, 1, 1).detrend_polynomial(degree=5).data
    # assert_array_almost_equal(output, target['detrend_polynomial_deg=5'])

def test_has_missing_points(data):
    surface = Surface(data, 1, 1)
    assert surface.has_missing_points == False
    surface.data[0,0] = np.nan
    assert surface.has_missing_points == True
    surface.data[0, 0] = None
    assert surface.has_missing_points == True

def test_zero(surface):
    assert surface.zero().data.min() == pytest.approx(0)
    assert surface.zero().data.max() == pytest.approx(surface.data.max() - surface.data.min())#

def test_center(surface):
    assert (surface + 5).center().data.mean() == pytest.approx(0)
    assert (surface + 100).center().data.mean() == pytest.approx(0)
    assert (surface - 100).center().data.mean() == pytest.approx(0)

def test_remove_outliers(noisy_surface):
    assert np.isnan(noisy_surface.remove_outliers(n=3).data.max())
    assert np.nanmax(noisy_surface.remove_outliers(n=3).data) == pytest.approx(1.788910463)
    assert np.nanmax(noisy_surface.remove_outliers(n=2).data) == pytest.approx(1.498192507)
    assert np.nanmax(noisy_surface.remove_outliers(n=1).data) == pytest.approx(0.749571902)

def test_threshold(noisy_surface):
    thresholded_surface = noisy_surface.threshold(0.5)
    assert np.nanmin(thresholded_surface.data) == pytest.approx(-1.496382184)
    assert np.isnan(np.max(thresholded_surface.data))
    assert np.nanmax(thresholded_surface.data) == pytest.approx(1.342712460)

    thresholded_surface = noisy_surface.threshold((0.5, 2))
    assert np.isnan(np.min(thresholded_surface.data))
    assert np.isnan(np.max(thresholded_surface.data))
    assert np.nanmax(thresholded_surface.data) == pytest.approx(1.34271246)
    assert np.nanmin(thresholded_surface.data) == pytest.approx(-1.3238377)

def test_fill_nonmeasured(noisy_surface):
    surface_with_missing_points = noisy_surface.remove_outliers()
    assert not bool(np.any(np.isnan(surface_with_missing_points.fill_nonmeasured().data)))
    assert np.max(surface_with_missing_points.fill_nonmeasured().data) == pytest.approx(1.7889104638)