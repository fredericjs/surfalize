from surfalize import Surface
import numpy as np
import pytest

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
    assert surface.Sdr() == pytest.approx(276.112693, abs=EPSILON)


def test_Sdq(surface):
    assert surface.Sdq() == pytest.approx(4.079605, abs=EPSILON)


def test_Sal(surface):
    assert surface.Sal() == pytest.approx(4.274174, abs=EPSILON)


def test_Str(surface):
    assert surface.Str() == pytest.approx(0.121940, abs=EPSILON)


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
    assert surface.Vmp() == pytest.approx(0.014809, abs=EPSILON)


def test_Vmc(surface):
    assert surface.Vmc() == pytest.approx(0.797036, abs=EPSILON)


def test_Vvv(surface):
    assert surface.Vvv() == pytest.approx(0.056284, abs=EPSILON)


def test_Vvc(surface):
    assert surface.Vvc() == pytest.approx(0.924875, abs=EPSILON)


def test_period(surface):
    assert surface.period() == pytest.approx(19.98, abs=EPSILON)


def test_depth(surface):
    assert surface.depth() == pytest.approx((1.870294, 0.058398), abs=EPSILON)


def test_aspect_ratio(surface):
    assert surface.aspect_ratio() == pytest.approx(0.093608, abs=EPSILON)


def test_homogeneity(surface):
    assert surface.homogeneity() == pytest.approx(0.9985, abs=EPSILON)


# New ISO 25178-2:2021 parameters #####################################################################################

def test_Spkx(surface):
    assert surface.Spkx() == pytest.approx(0.651521, abs=EPSILON)

def test_Svkx(surface):
    assert surface.Svkx() == pytest.approx(0.887011, abs=EPSILON)

def test_Sak1(surface):
    assert surface.Sak1() == pytest.approx(0.197043, abs=EPSILON)

def test_Sak2(surface):
    assert surface.Sak2() == pytest.approx(1.090266, abs=EPSILON)

def test_Sdc(surface):
    assert surface.Sdc() == pytest.approx(1.168843, abs=EPSILON)

def test_Ssw(surface):
    assert surface.Ssw() == pytest.approx(19.98, abs=1e-2)

def test_Sak1_equals_triangle_area(surface):
    assert surface.Sak1() == pytest.approx(0.5 * surface.Spk() * surface.Smr1(), abs=EPSILON)

def test_Sak2_equals_triangle_area(surface):
    assert surface.Sak2() == pytest.approx(0.5 * surface.Svk() * (100 - surface.Smr2()), abs=EPSILON)

def test_Sdc_equals_Sxp_at_defaults(surface):
    assert surface.Sdc(2.5, 50) == pytest.approx(surface.Sxp(2.5, 50), abs=EPSILON)

def test_Smrk_aliases(surface):
    assert surface.Smrk1() == surface.Smr1()
    assert surface.Smrk2() == surface.Smr2()

def test_Spkx_geq_Spk(surface):
    assert surface.Spkx() >= surface.Spk()
    assert surface.Svkx() >= surface.Svk()

def test_general_volume_parameters_reduce_to_special_cases(surface):
    # Vmp/Vmc/Vvv/Vvc are special cases of the general Vm(p)/Vv(p)
    assert surface.Vmp() == pytest.approx(surface.Vm(10), abs=EPSILON)
    assert surface.Vvv() == pytest.approx(surface.Vv(80), abs=EPSILON)
    assert surface.Vmc() == pytest.approx(surface.Vm(80) - surface.Vm(10), abs=EPSILON)
    assert surface.Vvc() == pytest.approx(surface.Vv(10) - surface.Vv(80), abs=EPSILON)


def test_size(surface):
    size = surface.size
    assert size.x == surface.data.shape[1]
    assert size.y == surface.data.shape[0]


# Analytic anchors ####################################################################################################
# A surface that is sinusoidal along x and constant along y, z(x) = a*sin(2*pi*x/lambda), sampled over an integer
# number of periods has closed-form areal height parameters, which anchors correctness rather than guarding change.
ANALYTIC_AMPLITUDE = 2.0

@pytest.fixture
def sinusoidal_surface():
    # samples_per_period divisible by 4 so the peak/valley land exactly on a sample
    samples_per_period = 200
    n_periods = 10
    nx = samples_per_period * n_periods
    ny = 100
    x = np.arange(nx)
    row = ANALYTIC_AMPLITUDE * np.sin(2 * np.pi * x / samples_per_period)
    data = np.tile(row, (ny, 1))
    return Surface(data, 0.1, 0.1)

def test_analytic_Sa(sinusoidal_surface):
    assert sinusoidal_surface.Sa() == pytest.approx(2 / np.pi * ANALYTIC_AMPLITUDE, abs=1e-3)

def test_analytic_Sq(sinusoidal_surface):
    assert sinusoidal_surface.Sq() == pytest.approx(ANALYTIC_AMPLITUDE / np.sqrt(2), abs=1e-3)

def test_analytic_Ssk(sinusoidal_surface):
    assert sinusoidal_surface.Ssk() == pytest.approx(0, abs=1e-6)

def test_analytic_Sku(sinusoidal_surface):
    assert sinusoidal_surface.Sku() == pytest.approx(1.5, abs=1e-3)

def test_analytic_Sp_Sv_Sz(sinusoidal_surface):
    assert sinusoidal_surface.Sp() == pytest.approx(ANALYTIC_AMPLITUDE, abs=1e-3)
    assert sinusoidal_surface.Sv() == pytest.approx(ANALYTIC_AMPLITUDE, abs=1e-3)
    assert sinusoidal_surface.Sz() == pytest.approx(2 * ANALYTIC_AMPLITUDE, abs=1e-3)

def test_analytic_period(sinusoidal_surface):
    # lambda = samples_per_period * step = 200 * 0.1 = 20.0
    assert sinusoidal_surface.period() == pytest.approx(20.0, abs=1e-2)

def test_available_parameters_are_callable():
    # Guards against registry drift: every registered parameter name must resolve to a callable method on the class.
    for name in Surface.AVAILABLE_PARAMETERS:
        assert callable(getattr(Surface, name, None)), f'{name!r} is registered but not a callable method'


def test_iso_parameters_subset_of_available():
    assert set(Surface.ISO_PARAMETERS) <= set(Surface.AVAILABLE_PARAMETERS)
