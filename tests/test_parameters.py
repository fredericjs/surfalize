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
    assert surface.Sal() == pytest.approx(4.347147, abs=EPSILON)


def test_Str(surface):
    assert surface.Str() == pytest.approx(0.062759, abs=EPSILON)


def test_Sk(surface):
    assert surface.Sk() == pytest.approx(2.207543, abs=EPSILON)


def test_Spk(surface):
    assert surface.Spk() == pytest.approx(0.190107, abs=EPSILON)


def test_Svk(surface):
    assert surface.Svk() == pytest.approx(0.398828, abs=EPSILON)


def test_Smr1(surface):
    assert surface.Smr1() == pytest.approx(2.009438, abs=EPSILON)


def test_Smr2(surface):
    assert surface.Smr2() == pytest.approx(89.979619, abs=EPSILON)


def test_Sxp(surface):
    assert surface.Sxp() == pytest.approx(1.168894, abs=EPSILON)


def test_Vmp(surface):
    assert surface.Vmp() == pytest.approx(0.014831, abs=EPSILON)


def test_Vmc(surface):
    assert surface.Vmc() == pytest.approx(0.797087, abs=EPSILON)


def test_Vvv(surface):
    assert surface.Vvv() == pytest.approx(0.056308, abs=EPSILON)


def test_Vvc(surface):
    assert surface.Vvc() == pytest.approx(0.924853, abs=EPSILON)


def test_period(surface):
    assert surface.period() == pytest.approx(19.98, abs=EPSILON)


def test_depth(surface):
    assert surface.depth() == pytest.approx((1.870294, 0.058398), abs=EPSILON)


def test_aspect_ratio(surface):
    assert surface.aspect_ratio() == pytest.approx(0.093608, abs=EPSILON)


def test_homogeneity(surface):
    assert surface.homogeneity() == pytest.approx(0.9985, abs=EPSILON)


# Orientation and texture direction ###################################################################################

@pytest.fixture
def grooved_surface():
    # Sinusoidal grooves with wavefronts rotated by 30 degrees
    theta = np.deg2rad(30)
    n = 400
    period_px = 40
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    z = np.sin(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / period_px)
    return Surface(z, 0.1, 0.1)

def test_orientation_fft(grooved_surface):
    # The fft method recovers the groove angle (magnitude) within a few degrees
    assert abs(grooved_surface.orientation(method='fft')) == pytest.approx(30, abs=2)

def test_orientation_fft_refined(grooved_surface):
    # The refined method (the default) recovers the angle more precisely
    assert abs(grooved_surface.orientation(method='fft_refined')) == pytest.approx(30, abs=1)

def test_orientation_default_is_refined(grooved_surface):
    assert grooved_surface.orientation() == grooved_surface.orientation(method='fft_refined')

def test_orientation_invalid_method(grooved_surface):
    with pytest.raises(ValueError):
        grooved_surface.orientation(method='nonsense')

def test_Std(grooved_surface):
    # Texture direction is the angle of maximum angular amplitude density
    assert grooved_surface.Std() == pytest.approx(30, abs=2)


# New ISO 25178-2:2021 parameters #####################################################################################

def test_Spkx(surface):
    assert surface.Spkx() == pytest.approx(0.648443, abs=EPSILON)

def test_Svkx(surface):
    assert surface.Svkx() == pytest.approx(0.997632, abs=EPSILON)

def test_Sak1(surface):
    assert surface.Sak1() == pytest.approx(0.191004, abs=EPSILON)

def test_Sak2(surface):
    assert surface.Sak2() == pytest.approx(1.998202, abs=EPSILON)

def test_Sdc(surface):
    assert surface.Sdc() == pytest.approx(1.168894, abs=EPSILON)

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

# Feature parameters (ISO 25178-2:2021, clause 5) ####################################################################

@pytest.fixture
def bump_surface():
    # A regular 4x3 grid of identical, well-separated Gaussian bumps on a flat field. Each bump is one hill with one
    # peak, so the significant-feature counts are known exactly. Bumps are kept away from the border so the peak
    # curvature can be evaluated, and the field is noise-free so Wolf pruning keeps exactly the twelve bumps. The bump
    # counts below use exclude_edge=False, since every bump in this small grid lies in the outer ring.
    step = 0.1
    ny, nx = 300, 400
    amplitude = 2.0
    sigma_px = 8.0
    y, x = np.mgrid[0:ny, 0:nx]
    z = np.zeros((ny, nx))
    centers = [(cy, cx) for cy in range(60, ny, 90) for cx in range(60, nx, 90)]
    for cy, cx in centers:
        z += amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_px ** 2))
    surface = Surface(z, step, step)
    surface._n_bumps = len(centers)
    surface._sigma_um = sigma_px * step
    surface._amplitude = amplitude
    return surface


def test_feature_peak_count(bump_surface):
    # Spd is the number of significant hills (peaks) per unit area; recovering the count recovers each bump.
    count = round(bump_surface.Spd(exclude_edge=False) * bump_surface.width_um * bump_surface.height_um)
    assert count == bump_surface._n_bumps


def test_feature_density_units(bump_surface):
    # Density is count divided by the evaluation area, in 1/µm².
    expected = bump_surface._n_bumps / (bump_surface.width_um * bump_surface.height_um)
    assert bump_surface.Spd(exclude_edge=False) == pytest.approx(expected, rel=1e-9)


def test_feature_edge_exclusion_reduces_count(bump_surface):
    # Excluding incomplete (border-touching) motifs can only reduce the count. Every bump in this grid lies in the
    # outer ring, so edge exclusion removes them all.
    assert bump_surface.Spd(exclude_edge=True) < bump_surface.Spd(exclude_edge=False)


def test_feature_peak_curvature_sign(bump_surface):
    # Peaks are convex (Spc > 0); the surrounding dale is concave (Svc < 0).
    assert bump_surface.Spc(exclude_edge=False) > 0
    assert bump_surface.Svc(exclude_edge=False) < 0


def test_feature_peak_curvature_value(bump_surface):
    # For a Gaussian bump A*exp(-r²/2σ²) the mean curvature at the apex is A/σ², which the discrete second difference
    # recovers to within a few percent.
    expected = bump_surface._amplitude / bump_surface._sigma_um ** 2
    assert bump_surface.Spc(exclude_edge=False) == pytest.approx(expected, rel=0.05)


def test_feature_five_point_peak_height(bump_surface):
    # All bumps are identical, so the five-point peak height equals the peak height of a single bump above the mean
    # plane.
    peak_height = bump_surface._amplitude - bump_surface.data.mean()
    assert bump_surface.S5p(exclude_edge=False) == pytest.approx(peak_height, rel=1e-3)


def test_feature_ten_point_height_is_sum(bump_surface):
    assert bump_surface.S10z(exclude_edge=False) == pytest.approx(
        bump_surface.S5p(exclude_edge=False) + bump_surface.S5v(exclude_edge=False), abs=EPSILON)


def test_feature_pruning_is_monotonic(bump_surface):
    # A larger pruning threshold can only remove features, never add them.
    assert (bump_surface.Spd(pruning=20, exclude_edge=False)
            <= bump_surface.Spd(pruning=5, exclude_edge=False))


def test_feature_parameters_match_mountainsmap():
    # Regression anchor against MountainsMap on the separable surface z = cos(x/3) + sin(y/3). The reference values
    # were taken from MountainsMap (no S-filter, no F-operation) in both edge modes.
    x = np.linspace(0, 94.399452, 2048)
    y = np.linspace(0, 70.78806, 1536)
    X, Y = np.meshgrid(x, y)
    surface = Surface(np.cos(X / 3) + np.sin(Y / 3), 0.046116, 0.046116)
    # Edge motifs excluded (MountainsMap default).
    assert surface.Spd() == pytest.approx(0.001197, abs=1e-6)
    assert surface.Svd() == pytest.approx(0.001347, abs=1e-6)
    # Edge motifs included.
    assert surface.Spd(exclude_edge=False) == pytest.approx(0.003592, abs=1e-6)
    assert surface.Svd(exclude_edge=False) == pytest.approx(0.003741, abs=1e-6)
    # Edge-independent parameters.
    assert surface.Spc() == pytest.approx(0.1111, abs=1e-3)
    assert surface.Svc() == pytest.approx(-0.1111, abs=1e-3)
    assert surface.S5p() == pytest.approx(1.957, abs=1e-3)
    assert surface.S5v() == pytest.approx(2.043, abs=1e-3)
    assert surface.S10z() == pytest.approx(4.000, abs=1e-3)


def test_feature_parameters_require_filled_surface():
    data = np.ones((10, 10))
    data[0, 0] = np.nan
    surface = Surface(data, 1.0, 1.0)
    with pytest.raises(ValueError):
        surface.Spd()


def test_feature_parameters_in_roughness_parameters(bump_surface):
    results = bump_surface.roughness_parameters(['Spd', 'S5p', 'S10z'])
    assert set(results.keys()) == {'Spd', 'S5p', 'S10z'}


def test_available_parameters_are_callable():
    # Guards against registry drift: every registered parameter name must resolve to a callable method on the class.
    for name in Surface.AVAILABLE_PARAMETERS:
        assert callable(getattr(Surface, name, None)), f'{name!r} is registered but not a callable method'


def test_iso_parameters_subset_of_available():
    assert set(Surface.ISO_PARAMETERS) <= set(Surface.AVAILABLE_PARAMETERS)
