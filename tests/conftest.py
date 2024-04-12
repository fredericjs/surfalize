import pytest
import numpy as np
from surfalize import Surface

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
    z = np.sin(x / period * 2 * np.pi)
    z += 0.5 * (np.sin((x - period / 2) / period * 2 * np.pi) + 1) * 0.3 * np.sin(
        (np.sin(y / 10) + x) / period_lipss * 2 * np.pi)
    z += np.random.normal(size=z.shape) / 5
    surface = Surface(z, 0.1, 0.1)
    return surface