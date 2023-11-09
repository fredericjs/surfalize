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
    assert surface.Sa().round(8) == 0.63137515

def test_sq():
    assert surface.Sq().round(8) == 0.70710678

def test_sz():
    assert surface.Sz().round(8) == 2.0
y
def test_sv():
    assert surface.Sv().round(8) == 1.0

def test_sp():
    assert surface.Sp().round(8) == 1

def test_ssk():
    assert surface.Ssk().round(8) == 0.0

def test_sku():
    assert surface.Sku().round(8) == 1.50000001

def test_sdr():
    assert round(surface.Sdr(), 8) == 2.6163234

def test_period():
    assert round(surface.period(), 8) == period

def test_depth():
    depth, std = surface.depth(retstd=True)
    assert round(depth, 8) == 1.90211303
    assert round(std, 8) == 0.0

