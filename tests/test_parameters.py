from surfalize import Surface
import numpy as np
import pytest

@pytest.fixture
def surface():
    np.random.seed(0)
    step = 0.1
    period = 20 / step
    period_lipss = 1 / step
    nx = 1000
    ny = 700
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y)
    z = np.sin(x / period * 2 * np.pi) + 10
    z += 0.5 * (np.sin((x - period / 2) / period * 2 * np.pi) + 1) * 0.3 * np.sin(
        (np.sin(y / 10) + x) / period_lipss * 2 * np.pi)
    z += np.random.normal(size=z.shape) / 5
    surface = Surface(z, 0.1, 0.1)
    return surface

EPSILON = 1e-6

def test_Sa(surface):
    assert surface.Sa() == pytest.approx(0.653132, abs=EPSILON)

def test_Sq(surface):
    assert surface.Sq() == pytest.approx(0.745961, abs=EPSILON)

def test_Sp(surface):
    assert surface.Sp() == pytest.approx(1.854956, abs=EPSILON)

def test_Sv(surface):
    assert surface.Sv() == pytest.approx(1.998661, abs=EPSILON)

def test_Sz(surface):
    assert surface.Sz() == pytest.approx(3.853617, abs=EPSILON)

def test_Ssk(surface):
    assert surface.Ssk() == pytest.approx(-0.08144, abs=EPSILON)

def test_Sku(surface):
    assert surface.Sku() == pytest.approx(1.818014, abs=EPSILON)

def test_Sdr(surface):
    assert surface.Sdr() == pytest.approx(276.103672, abs=EPSILON)

def test_Sdq(surface):
    assert surface.Sdq() == pytest.approx(4.079605, abs=EPSILON)

def test_Sal(surface):
    assert surface.Sal() == pytest.approx(4.35, abs=EPSILON)

def test_Str(surface):
    assert surface.Str() == pytest.approx(0.123845, abs=EPSILON)

def test_Sk(surface):
    assert surface.Sk() == pytest.approx(2.315085, abs=EPSILON)

def test_Spk(surface):
    assert surface.Spk() == pytest.approx(0.191242, abs=EPSILON)

def test_Svk(surface):
    assert surface.Svk() == pytest.approx(0.335035, abs=EPSILON)

def test_Smr1(surface):
    assert surface.Smr1() == pytest.approx(2.060661, abs=EPSILON)

def test_Smr2(surface):
    assert surface.Smr2() == pytest.approx(93.491628, abs=EPSILON)

def test_Sxp(surface):
    assert surface.Sxp() == pytest.approx(1.168843, abs=EPSILON)

def test_Vmp(surface):
    assert surface.Vmp() == pytest.approx(-0.014809, abs=EPSILON)

def test_Vmc(surface):
    assert surface.Vmc() == pytest.approx(-0.797036, abs=EPSILON)

def test_Vvv(surface):
    assert surface.Vvv() == pytest.approx(0.056284, abs=EPSILON)

def test_Vvc(surface):
    assert surface.Vvc() == pytest.approx(0.924875, abs=EPSILON)

def test_period(surface):
    assert surface.period() == pytest.approx(19.98, abs=EPSILON)

def test_depth(surface):
    assert surface.depth() == pytest.approx((1.876099, 0.060096), abs=EPSILON)

def test_aspect_ratio(surface):
    assert surface.aspect_ratio() == pytest.approx(0.093899, abs=EPSILON)

def test_homogeneity(surface):
    assert surface.homogeneity() == pytest.approx(0.9986, abs=EPSILON)

def test_size(surface):
    size = surface.size
    assert size.x == surface.data.shape[1]
    assert size.y == surface.data.shape[0]

