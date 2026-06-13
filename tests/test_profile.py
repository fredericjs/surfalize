import numpy as np
import pytest
from surfalize import Profile

EPSILON = 1e-6

@pytest.fixture
def profile():
    np.random.seed(0)
    x = np.arange(1000)
    data = np.sin(x / 50 * 2 * np.pi) + np.random.normal(size=1000) * 0.1
    return Profile(data, 0.1)

@pytest.fixture
def noisy_profile(profile):
    np.random.seed(1)
    mask = (np.random.normal(size=profile.size) > 2.5).astype('float')
    return Profile(profile.data + mask * 10, profile.step)

# Height parameters ####################################################################################################

def test_Ra(profile):
    assert profile.Ra() == pytest.approx(0.638341, abs=EPSILON)

def test_Rq(profile):
    assert profile.Rq() == pytest.approx(0.713584, abs=EPSILON)

def test_Rp(profile):
    assert profile.Rp() == pytest.approx(1.146977, abs=EPSILON)

def test_Rv(profile):
    assert profile.Rv() == pytest.approx(1.179609, abs=EPSILON)

def test_Rz(profile):
    assert profile.Rz() == pytest.approx(2.326586, abs=EPSILON)

def test_Rz_single_section_equals_Rt(profile):
    assert profile.Rz(n_sections=1) == pytest.approx(profile.Rt(), abs=EPSILON)

def test_Rt(profile):
    assert profile.Rt() == pytest.approx(2.479203, abs=EPSILON)

def test_Rsk(profile):
    assert profile.Rsk() == pytest.approx(-0.002984, abs=EPSILON)

def test_Rku(profile):
    assert profile.Rku() == pytest.approx(1.564224, abs=EPSILON)

def test_Rdq(profile):
    assert profile.Rdq() == pytest.approx(1.667171, abs=EPSILON)

# Functional parameters ################################################################################################

def test_Rk(profile):
    assert profile.Rk() == pytest.approx(2.021790, abs=EPSILON)

def test_Rpk(profile):
    assert profile.Rpk() == pytest.approx(0.070079, abs=EPSILON)

def test_Rvk(profile):
    assert profile.Rvk() == pytest.approx(0.208167, abs=EPSILON)

def test_Rmr1(profile):
    assert profile.Rmr1() == pytest.approx(0.9, abs=EPSILON)

def test_Rmr2(profile):
    assert profile.Rmr2() == pytest.approx(86.8, abs=EPSILON)

def test_Rxp(profile):
    assert profile.Rxp() == pytest.approx(1.084652, abs=EPSILON)

def test_Vmp(profile):
    assert profile.Vmp() == pytest.approx(0.008390, abs=EPSILON)

def test_Vmc(profile):
    assert profile.Vmc() == pytest.approx(0.826045, abs=EPSILON)

def test_Vvv(profile):
    assert profile.Vvv() == pytest.approx(0.029502, abs=EPSILON)

def test_Vvc(profile):
    assert profile.Vvc() == pytest.approx(0.921124, abs=EPSILON)

def test_period(profile):
    assert profile.period() == pytest.approx(5.0, abs=0.1)

def test_roughness_parameters(profile):
    results = profile.roughness_parameters()
    assert set(results.keys()) == set(Profile.ISO_PARAMETERS)
    assert results['Ra'] == pytest.approx(profile.Ra(), abs=EPSILON)

def test_roughness_parameters_invalid(profile):
    with pytest.raises(ValueError):
        profile.roughness_parameters(['Sa'])

# Operations ###########################################################################################################

def test_center(profile):
    assert profile.center().data.mean() == pytest.approx(0)
    assert Profile(profile.data + 100, profile.step).center().data.mean() == pytest.approx(0)

def test_zero(profile):
    assert profile.zero().data.min() == pytest.approx(0)
    assert profile.zero().data.max() == pytest.approx(profile.data.max() - profile.data.min())

def test_invert(profile):
    inverted = profile.invert()
    assert inverted.data.max() == pytest.approx(profile.data.max())
    assert inverted.data.min() == pytest.approx(profile.data.min())
    assert np.argmax(inverted.data) == np.argmin(profile.data)

def test_remove_outliers(noisy_profile):
    assert np.isnan(noisy_profile.remove_outliers(n=3).data.max())
    assert np.nanmax(noisy_profile.remove_outliers(n=3).data) < 10

def test_threshold(noisy_profile):
    thresholded = noisy_profile.threshold(0.5)
    assert thresholded.has_missing_points
    assert np.nanmax(thresholded.data) < noisy_profile.data.max()

def test_fill_nonmeasured(noisy_profile):
    profile_with_missing_points = noisy_profile.remove_outliers()
    assert profile_with_missing_points.has_missing_points
    filled = profile_with_missing_points.fill_nonmeasured()
    assert not filled.has_missing_points

def test_level():
    np.random.seed(0)
    data = np.random.uniform(size=1000)
    tilted = Profile(data + np.linspace(0, 5, 1000), 0.1)
    leveled = tilted.level()
    # The least squares line of the leveled profile should be flat
    x = np.linspace(-1, 1, leveled.size)
    slope = np.polynomial.polynomial.polyfit(x, leveled.data, 1)[1]
    assert slope == pytest.approx(0, abs=1e-8)

def test_detrend_polynomial():
    np.random.seed(0)
    x = np.linspace(-1, 1, 1000)
    data = np.random.uniform(size=1000) + 5 * x ** 2
    detrended = Profile(data, 0.1).detrend_polynomial(degree=2)
    coeffs = np.polynomial.polynomial.polyfit(x, detrended.data, 2)
    assert coeffs[2] == pytest.approx(0, abs=1e-8)

def test_detrend_return_trend(profile):
    detrended, trend = profile.detrend_polynomial(degree=1, return_trend=True)
    assert np.allclose(detrended.data + trend.data, profile.data)

def test_filter_lowpass(profile):
    filtered = profile.filter('lowpass', 5)
    assert isinstance(filtered, Profile)
    assert filtered.size == profile.size
    # Lowpass filtering should reduce the roughness
    assert filtered.Ra() < profile.Ra()

def test_filter_highpass(profile):
    filtered = profile.filter('highpass', 5)
    assert filtered.data.mean() == pytest.approx(0, abs=1e-3)

def test_filter_both(profile):
    high, low = profile.filter('both', 5)
    assert np.allclose(high.data + low.data, profile.data)

def test_filter_bandpass(profile):
    filtered = profile.filter('bandpass', 1, 20)
    assert isinstance(filtered, Profile)

def test_filter_inplace(profile):
    target = profile.filter('lowpass', 5)
    profile.filter('lowpass', 5, inplace=True)
    assert np.allclose(profile.data, target.data)

def test_crop(profile):
    cropped = profile.crop((10, 50))
    assert cropped.length_um == pytest.approx(40)
    assert cropped.size == 401
    with pytest.raises(ValueError):
        profile.crop((-1, 50))
    with pytest.raises(ValueError):
        profile.crop((0, profile.length_um + 1))

def test_crop_inplace(profile):
    target = profile.crop((10, 50))
    profile.crop((10, 50), inplace=True)
    assert np.allclose(profile.data, target.data)
    assert profile.length_um == target.length_um

def test_zoom(profile):
    zoomed = profile.zoom(2)
    assert zoomed.size == pytest.approx(profile.size / 2, abs=1)

def test_length_um_calculation():
    profile = Profile(np.zeros(101), 0.5)
    assert profile.length_um == pytest.approx(50)

def test_cache_invalidation_on_inplace_operation(profile):
    ra_before = profile.Ra()
    profile.crop((0, 50), inplace=True)
    assert profile.Ra() != ra_before

def test_size(profile):
    assert profile.size == profile.data.shape[0]

def test_has_missing_points(profile):
    assert profile.has_missing_points == False
    profile.data[0] = np.nan
    assert profile.has_missing_points == True

# Analytic anchors ####################################################################################################
# A pure sinusoid z(x) = a*sin(2*pi*x/lambda) sampled over an integer number of periods has closed-form roughness
# parameters, which anchors correctness rather than merely guarding against change.
AMPLITUDE = 2.0

@pytest.fixture
def sinusoidal_profile():
    # samples_per_period divisible by 4 so the peak/valley land exactly on a sample
    samples_per_period = 200
    n_periods = 10
    n = samples_per_period * n_periods
    x = np.arange(n)
    data = AMPLITUDE * np.sin(2 * np.pi * x / samples_per_period)
    return Profile(data, 0.1)

def test_analytic_Ra(sinusoidal_profile):
    # Ra = (2/pi) * a
    assert sinusoidal_profile.Ra() == pytest.approx(2 / np.pi * AMPLITUDE, abs=1e-3)

def test_analytic_Rq(sinusoidal_profile):
    # Rq = a / sqrt(2)
    assert sinusoidal_profile.Rq() == pytest.approx(AMPLITUDE / np.sqrt(2), abs=1e-3)

def test_analytic_Rsk(sinusoidal_profile):
    # Symmetric distribution -> skewness 0
    assert sinusoidal_profile.Rsk() == pytest.approx(0, abs=1e-6)

def test_analytic_Rku(sinusoidal_profile):
    # Kurtosis of a sinusoid = 3/2
    assert sinusoidal_profile.Rku() == pytest.approx(1.5, abs=1e-3)

def test_analytic_Rt(sinusoidal_profile):
    # Peak-to-valley = 2a
    assert sinusoidal_profile.Rt() == pytest.approx(2 * AMPLITUDE, abs=1e-3)

def test_analytic_Rz(sinusoidal_profile):
    # Each sampling section spans full periods, so section peak-to-valley = 2a
    assert sinusoidal_profile.Rz() == pytest.approx(2 * AMPLITUDE, abs=1e-3)

def test_analytic_Rp_Rv(sinusoidal_profile):
    assert sinusoidal_profile.Rp() == pytest.approx(AMPLITUDE, abs=1e-3)
    assert sinusoidal_profile.Rv() == pytest.approx(AMPLITUDE, abs=1e-3)

def test_available_parameters_are_callable():
    # Guards against registry drift: every registered parameter name must resolve to a callable method on the class.
    for name in Profile.AVAILABLE_PARAMETERS:
        assert callable(getattr(Profile, name, None)), f'{name!r} is registered but not a callable method'

def test_iso_parameters_subset_of_available():
    assert set(Profile.ISO_PARAMETERS) <= set(Profile.AVAILABLE_PARAMETERS)
