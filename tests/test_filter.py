import pytest
from surfalize import Surface
from surfalize.filter import GaussianFilter

@pytest.mark.parametrize('cutoff, expected_mean, expected_std', [
    (2, 0.00045879628758761483, 0.6829918577051731),
    (5, 0.0004587962875876147, 0.5750573603806917),
    (1.5, 0.00045879628758761423, 0.6933187088833069)
])
def test_filter_lowpass(surface, cutoff, expected_mean, expected_std):
    filter = GaussianFilter(cutoff, 'lowpass')
    filtered_surface = filter.apply(surface)
    assert filtered_surface.data.mean() == pytest.approx(expected_mean)
    assert filtered_surface.data.std() == pytest.approx(expected_std)