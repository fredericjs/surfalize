import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from surfalize.mathutils import argclosest, closest, interp1d, _sinusoid, Sinusoid, otsu_threshold

@pytest.fixture
def array():
    return np.array([1, 2, 3, 4, 5])
@pytest.fixture
def expected_sinusoid_result():
    return np.array([1., -1., 1., 3., 1.])

@pytest.fixture
def sinusoid_object():
    return Sinusoid(2, 2, -1, 1)

def test_argclosest(array):
    assert argclosest(2.2, array) == 1
    assert argclosest(4.6, array) == 4

def test_closest(array):
    assert closest(2.2, array) == 2
    assert closest(4.6, array) == 5

def test_interp1d(array):
    x = np.linspace(0, 10)
    y = np.linspace(20, 0)
    f = interp1d(x, y)
    f2 = interp1d(x, y[::-1])
    assert hasattr(f, '__call__')
    assert float(f(0)) ==  pytest.approx(20)
    assert float(f(10)) == pytest.approx(0)
    assert float(f(1)) == pytest.approx(18)
    assert float(f2(0)) == pytest.approx(0)
    assert float(f2(10)) == pytest.approx(20)
    assert float(f2(1)) == pytest.approx(2)

def test_otsu_threshold():
    # Two well-separated clusters around 0 and 10 -> threshold must fall between them
    values = np.concatenate([np.full(500, 0.0), np.full(500, 10.0)])
    threshold = otsu_threshold(values)
    assert 0 < threshold < 10
    # The threshold must correctly separate the two clusters
    assert (values > threshold).sum() == 500
    # Non-finite values must be ignored
    values_with_nan = np.concatenate([values, np.full(50, np.nan)])
    assert otsu_threshold(values_with_nan) == pytest.approx(threshold)

def test_sinusoid(expected_sinusoid_result):
    assert _sinusoid(0, 1, 2, 0, 0) == pytest.approx(0)
    assert _sinusoid(2, 1, 2, 0, 0) == pytest.approx(0)
    assert _sinusoid(0.5, 2, 2, 0, 0) == pytest.approx(2)
    arr = _sinusoid(np.linspace(0, 2, 5), 2, 2, -1, 1)
    assert_array_almost_equal(arr, expected_sinusoid_result)

class TestSinusoid:
    def setup_method(self, method):
        self.sinusoid = Sinusoid(2, 2, -1, 1)

    def test_Sinusoid(self, expected_sinusoid_result):
        assert_array_almost_equal(self.sinusoid(np.linspace(0, 2, 5)), expected_sinusoid_result)

    def test_first_peak(self):
       assert self.sinusoid.first_peak() == pytest.approx(1.5)

    def test_first_extremum(self):
        assert self.sinusoid.first_extremum() == pytest.approx(0.5)

    def test_from_fit(self):
        # Use a local generator so the test is independent of the global RNG state and test execution order.
        # The fit must recover the true parameters of the generating sinusoid within a tolerance set by the noise.
        rng = np.random.default_rng(0)
        a, p, x0, y0 = 5, 0.5, 1, -0.5
        xdata = np.linspace(0, 10, 1000)
        ydata = a * np.sin((xdata - x0) / p * 2 * np.pi) + y0 + rng.normal(size=xdata.size)
        sinusoid = Sinusoid.from_fit(xdata, ydata, infer_p0=True)
        assert sinusoid.amplitude == pytest.approx(a, abs=0.2)
        assert sinusoid.period == pytest.approx(p, abs=0.05)
        assert sinusoid.y0 == pytest.approx(y0, abs=0.1)
        # x0 is only defined modulo the period; the true offset x0=1 is equivalent to 0 for p=0.5
        phase = sinusoid.x0 % sinusoid.period
        assert min(phase, sinusoid.period - phase) == pytest.approx(0, abs=0.05)