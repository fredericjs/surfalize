import numpy as np
import pytest
from surfalize.utils import argclosest, closest, interp1d, is_list_like

@pytest.fixture
def array():
    return np.array([1, 2, 3, 4, 5])

def test_is_list_like():
    assert is_list_like([])
    assert is_list_like((1, 2))
    assert is_list_like(np.array([1, 2, 3]))
    assert not is_list_like(1)
    assert not is_list_like(range(4))

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