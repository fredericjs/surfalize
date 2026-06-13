import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from surfalize import Surface, Profile

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
    surface[0, 0] = np.nan
    assert surface.has_missing_points == True
    surface[0, 0] = None
    assert surface.has_missing_points == True

def test_data_is_read_only(surface):
    with pytest.raises(ValueError):
        surface.data[0, 0] = 5

def test_setitem_clears_cache(surface):
    sa_before = surface.Sa()
    new_data = np.tile(np.linspace(-1, 1, surface.size.x), (surface.size.y, 1))
    surface[:, :] = new_data
    assert surface.Sa() != sa_before
    assert surface.Sa() == pytest.approx(Surface(new_data, surface.step_x, surface.step_y).Sa())

def test_data_setter_clears_cache_and_recomputes_dims(surface):
    sa_before = surface.Sa()
    new_data = np.tile(np.linspace(-1, 1, surface.size.x), (surface.size.y, 1))
    surface.data = new_data
    assert surface.Sa() != sa_before
    assert surface.Sa() == pytest.approx(Surface(new_data, surface.step_x, surface.step_y).Sa())
    assert surface.width_um == pytest.approx((surface.size.x - 1) * surface.step_x)

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

def test_eq(surface):
    assert surface == Surface(surface.data.copy(), surface.step_x, surface.step_y)
    # A surface that is everywhere lower must not compare equal
    assert surface != surface - 1
    assert surface != surface + 1
    # A surface with a different stepsize must not compare equal (scale both axes to avoid the unequal-pixel warning)
    assert surface != Surface(surface.data.copy(), surface.step_x * 2, surface.step_y * 2)

def test_rsub(surface):
    assert np.allclose((5 - surface).data, 5 - surface.data)
    assert np.allclose((surface - 5).data, surface.data - 5)

def test_eq_nan_handling(surface):
    # A NaN where the other surface has a finite value must not compare equal
    data = surface.data.copy()
    data[0, 0] = np.nan
    with_nan = Surface(data, surface.step_x, surface.step_y)
    assert surface != with_nan
    # Two surfaces with NaN at the same position and identical finite values compare equal
    data2 = surface.data.copy()
    data2[0, 0] = np.nan
    with_nan2 = Surface(data2, surface.step_x, surface.step_y)
    assert with_nan == with_nan2
    # A NaN at a different position must not compare equal
    data3 = surface.data.copy()
    data3[0, 1] = np.nan
    with_nan3 = Surface(data3, surface.step_x, surface.step_y)
    assert with_nan != with_nan3

@pytest.fixture
def stepheight_surface():
    # Flat upper surface at height 5 with a central rectangular cavity at height 0
    np.random.seed(0)
    data = np.full((200, 200), 5.0)
    data[50:150, 50:150] = 0.0
    data += np.random.normal(scale=0.01, size=data.shape)
    return Surface(data, 1, 1)

def test_stepheight(stepheight_surface):
    assert stepheight_surface.stepheight() == pytest.approx(5.0, abs=0.05)

def test_stepheight_mask_segments_two_levels(stepheight_surface):
    mask = stepheight_surface._stepheight_get_mask()
    # The upper level covers the whole surface except the 100x100 cavity
    assert mask.sum() == pytest.approx(200 * 200 - 100 * 100, abs=10)
    # The upper level (mask True) must have the higher mean height
    assert stepheight_surface.data[mask].mean() > stepheight_surface.data[~mask].mean()

def test_cavity_volume(stepheight_surface):
    # Cavity is 100x100 px at depth 5 -> volume approx 100*100*5
    assert stepheight_surface.cavity_volume() == pytest.approx(100 * 100 * 5, rel=0.02)

def test_stepheight_level(stepheight_surface):
    # Add a tilt that stepheight_level should remove based on the upper-level plane
    yy, xx = np.mgrid[0:stepheight_surface.size.y, 0:stepheight_surface.size.x]
    tilted = Surface(stepheight_surface.data + 0.01 * xx + 0.02 * yy,
                     stepheight_surface.step_x, stepheight_surface.step_y)
    leveled = tilted.stepheight_level()
    assert isinstance(leveled, Surface)
    # The upper level must become flat (its tilt removed) while the step is preserved
    mask = leveled._stepheight_get_mask()
    assert leveled.data[mask].std() < 0.05
    assert leveled.stepheight() == pytest.approx(5.0, abs=0.05)

# Geometric operations ################################################################################################

def test_crop(surface):
    cropped = surface.crop((10, 50, 5, 25))
    assert cropped.width_um == pytest.approx(40)
    assert cropped.height_um == pytest.approx(20)
    with pytest.raises(ValueError):
        surface.crop((-1, 50, 0, 10))

def test_crop_inplace(surface):
    target = surface.crop((10, 50, 5, 25))
    surface.crop((10, 50, 5, 25), inplace=True)
    assert surface == target

def test_zoom(surface):
    zoomed = surface.zoom(2)
    assert zoomed.width_um == pytest.approx(surface.width_um / 2, abs=surface.step_x)
    assert zoomed.height_um == pytest.approx(surface.height_um / 2, abs=surface.step_y)

def test_getitem(surface):
    sub = surface[0:50, 0:100]
    assert sub.size.y == 50
    assert sub.size.x == 100
    assert sub.step_x == surface.step_x
    assert sub.step_y == surface.step_y

def test_getitem_step_scales_stepsize(surface):
    sub = surface[::2, ::2]
    assert sub.step_x == pytest.approx(2 * surface.step_x)
    assert sub.step_y == pytest.approx(2 * surface.step_y)

def test_rotate(surface):
    rotated = surface.rotate(30)
    assert isinstance(rotated, Surface)
    # Rotating and cropping to the inscribed rectangle reduces the area
    assert rotated.size.x * rotated.size.y < surface.size.x * surface.size.y
    assert not rotated.has_missing_points

@pytest.fixture
def grooved_surface():
    # Sinusoidal grooves with wavefronts rotated by 30 degrees
    theta = np.deg2rad(30)
    n = 400
    period_px = 40
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    z = np.sin(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / period_px)
    return Surface(z, 0.1, 0.1)

def test_align(grooved_surface):
    aligned = grooved_surface.align(axis='y')
    assert isinstance(aligned, Surface)
    # After aligning to the y-axis the dominant texture is vertical -> orientation near 0
    assert aligned.orientation(method='fft') == pytest.approx(0, abs=2)

# Profile extraction ##################################################################################################

def test_get_horizontal_profile(surface):
    profile = surface.get_horizontal_profile(surface.height_um / 2)
    assert isinstance(profile, Profile)
    assert profile.length_um == pytest.approx(surface.width_um)
    assert profile.size == surface.size.x

def test_get_vertical_profile(surface):
    profile = surface.get_vertical_profile(surface.width_um / 2)
    assert isinstance(profile, Profile)
    assert profile.length_um == pytest.approx(surface.height_um)
    assert profile.size == surface.size.y

def test_get_oblique_profile(surface):
    profile = surface.get_oblique_profile(0, 0, surface.width_um, surface.height_um)
    assert isinstance(profile, Profile)
    assert profile.length_um == pytest.approx(np.hypot(surface.width_um, surface.height_um), rel=0.02)

def test_get_horizontal_profile_out_of_bounds(surface):
    with pytest.raises(ValueError):
        surface.get_horizontal_profile(surface.height_um + 1)

def test_fill_nonmeasured(noisy_surface):
    surface_with_missing_points = noisy_surface.remove_outliers()
    assert not bool(np.any(np.isnan(surface_with_missing_points.fill_nonmeasured().data)))
    assert np.max(surface_with_missing_points.fill_nonmeasured().data) == pytest.approx(1.7889104638)