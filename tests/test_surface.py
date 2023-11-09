from surfalize import Surface
import numpy as np

period = 20
nx = 1000
ny = 700
x = np.arange(nx)
y = np.arange(ny)
x, y = np.meshgrid(x, y)
z = np.sin(x/period * 2 * np.pi)

surface = Surface(z, 1, 1, nx, ny)

def test_sa():
    assert surface.Sa() == 1

def test_sq():
    assert surface.Sq() == 1