import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from surfalize import Surface, Mask


@pytest.fixture
def flat_surface():
    # 10x10 surface, all zeros, step 1 µm
    return Surface(np.zeros((10, 10)), 1, 1)


@pytest.fixture
def ramp_surface():
    # Linear ramp in x so leveling has a well-defined trend
    y, x = np.indices((10, 10))
    return Surface(x.astype(float) * 2.0, 1, 1)


def test_mask_starts_empty(flat_surface):
    assert flat_surface.mask.is_empty
    assert not flat_surface.has_masked_points
    assert flat_surface.mask._array is None  # lazy: no allocation


def test_empty_mask_is_noop(flat_surface):
    # An empty mask must not change any analysis result
    data = np.random.default_rng(0).uniform(size=(10, 10))
    s = Surface(data, 1, 1)
    assert s.Sa() == pytest.approx(np.abs(data - data.mean()).mean())
    assert s.min() == data.min()
    assert s.max() == data.max()


def test_add_rectangle_pixels(flat_surface):
    result = flat_surface.mask.add_rectangle((0, 4, 0, 4), in_units=False)
    arr = result.mask.to_array()
    assert arr[0:5, 0:5].all()
    assert not arr[5:, 5:].any()
    assert result.has_masked_points


def test_operation_returns_copy_by_default(flat_surface):
    result = flat_surface.mask.add_rectangle((0, 4, 0, 4), in_units=False)
    assert isinstance(result, Surface)
    assert result is not flat_surface
    # original is untouched
    assert not flat_surface.has_masked_points
    assert result.has_masked_points


def test_inplace_returns_same_surface(flat_surface):
    result = flat_surface.mask.add_rectangle((0, 4, 0, 4), in_units=False, inplace=True)
    assert result is flat_surface
    assert flat_surface.has_masked_points


def test_slicing_api(flat_surface):
    flat_surface.mask[2:5, 1:3] = True
    arr = flat_surface.mask.to_array()
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:5, 1:3] = True
    assert_array_equal(arr, expected)


def test_boolean_indexing(flat_surface):
    s = Surface(np.arange(100).reshape(10, 10).astype(float), 1, 1)
    s.mask[s.data > 50] = True
    assert_array_equal(s.mask.to_array(), s.data > 50)


def test_add_circle(flat_surface):
    result = flat_surface.mask.add_circle((5, 5), 2, in_units=False)
    arr = result.mask.to_array()
    # center masked, far corner not
    assert arr[5, 5]
    assert not arr[0, 0]


def test_subtract_rectangle(flat_surface):
    s = flat_surface.mask.add_rectangle((0, 9, 0, 9), in_units=False)
    s = s.mask.subtract_rectangle((2, 4, 2, 4), in_units=False)
    arr = s.mask.to_array()
    assert not arr[2:5, 2:5].any()
    assert arr[0, 0]


def test_invert(flat_surface):
    flat_surface.mask[0:5, :] = True
    result = flat_surface.mask.invert()
    arr = result.mask.to_array()
    assert not arr[0:5, :].any()
    assert arr[5:, :].all()


def test_clear(flat_surface):
    flat_surface.mask[0:5, :] = True
    result = flat_surface.mask.clear()
    assert result.mask.is_empty
    # original still masked since clear defaults to a copy
    assert flat_surface.has_masked_points


def test_threshold_method():
    s = Surface(np.arange(100).reshape(10, 10).astype(float), 1, 1)
    result = s.mask.threshold(below=20, above=80)
    arr = result.mask.to_array()
    assert_array_equal(arr, (s.data < 20) | (s.data > 80))


def test_chaining_inplace(flat_surface):
    result = (flat_surface.mask.add_rectangle((0, 2, 0, 2), in_units=False, inplace=True)
              .mask.add_circle((8, 8), 1, in_units=False, inplace=True))
    assert result is flat_surface
    assert flat_surface.has_masked_points


def test_mask_excludes_from_statistics():
    rng = np.random.default_rng(1)
    data = rng.uniform(size=(10, 10))
    data[0, 0] = 100.0  # outlier
    s = Surface(data, 1, 1)
    assert s.max() == 100.0
    s.mask[0, 0] = True
    rest = data.copy()
    rest[0, 0] = np.nan
    assert s.max() == np.nanmax(rest)
    assert s.min() == np.nanmin(rest)
    assert s.mean() == pytest.approx(np.nanmean(rest))
    assert s.Sa() == pytest.approx(np.nanmean(np.abs(rest - np.nanmean(rest))))


def test_mask_excludes_from_leveling():
    # A ramp with a masked spike: the leveling fit should ignore the spike
    y, x = np.indices((20, 20))
    data = x.astype(float)
    spike = data.copy()
    spike[10, 10] = 1000.0
    s_clean = Surface(data.copy(), 1, 1).level()
    s_masked = Surface(spike, 1, 1)
    s_masked.mask[10, 10] = True
    s_masked_leveled = s_masked.level()
    # Everywhere except the spike, leveling should match the clean result closely
    diff = np.abs(s_masked_leveled.data - s_clean.data)
    diff[10, 10] = 0
    assert diff.max() < 1e-6


def test_cache_invalidation_on_mask_change():
    data = np.ones((10, 10))
    data[0, 0] = 100.0
    s = Surface(data, 1, 1)
    first = s.max()
    assert first == 100.0
    s.mask[0, 0] = True
    assert s.max() == 1.0  # would still be 100 if cache was stale


def test_grid_parameters_blocked_when_masked():
    y, x = np.indices((50, 50))
    s = Surface(np.sin(x / 5).astype(float), 1, 1)
    s.mask[0, 0] = True
    with pytest.raises(ValueError, match='masked'):
        s.Sdq()
    with pytest.raises(ValueError, match='masked'):
        s.surface_area()


def test_mask_propagates_through_crop():
    s = Surface(np.zeros((10, 10)), 1, 1)
    s.mask[2:4, 2:4] = True
    cropped = s.crop((0, 4, 0, 4), in_units=False)  # keep rows/cols 0-4, includes the masked region
    assert cropped.has_masked_points
    assert cropped.mask.to_array().shape == tuple(cropped.size)
    assert cropped.mask.to_array()[2:4, 2:4].all()


def test_mask_propagates_through_getitem():
    s = Surface(np.zeros((10, 10)), 1, 1)
    s.mask[0:5, 0:5] = True
    sub = s[0:5, 0:5]
    assert sub.mask.to_array().all()


def test_mask_propagates_through_zoom():
    s = Surface(np.zeros((10, 10)), 1, 1)
    s.mask[:] = True
    zoomed = s.zoom(2)
    assert zoomed.mask.to_array().all()


def test_data_setter_resets_mismatched_mask():
    s = Surface(np.zeros((10, 10)), 1, 1)
    s.mask[0, 0] = True
    with pytest.warns(UserWarning):
        s.data = np.zeros((5, 5))
    assert s.mask.is_empty


def test_constructor_accepts_mask():
    m = np.zeros((10, 10), dtype=bool)
    m[0, 0] = True
    s = Surface(np.ones((10, 10)), 1, 1, mask=m)
    assert s.has_masked_points
    assert s.mask.to_array()[0, 0]


def test_mask_shape_validation(flat_surface):
    with pytest.raises(ValueError, match='shape'):
        flat_surface.mask.set(np.zeros((3, 3), dtype=bool))
