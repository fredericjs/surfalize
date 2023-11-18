from surfalize import Surface
import numpy as np
from pytest import approx

np.random.seed(0)

period = 20
nx = 1000
ny = 700
x = np.arange(nx)
y = np.arange(ny)
x, y = np.meshgrid(x, y)
z = np.sin(x/period * 2 * np.pi)
surface = Surface(z, 0.1, 0.1, 0.1*nx, 0.1*ny)

period = 80
nx = 1000
ny = 700
x = np.arange(nx)
y = np.arange(ny)
x, y = np.meshgrid(x, y)
z = np.sin(x/period * 2 * np.pi)
z_noise = z + np.random.normal(size=z.shape) / 5
surface_noise = Surface(z_noise, 0.1, 0.1, 0.1*nx, 0.1*ny)


def test_Sa():
	assert surface.Sa() == approx(0.63137515)
	assert surface_noise.Sa() == approx(0.64857575)

def test_Sq():
	assert surface.Sq() == approx(0.70710678)
	assert surface_noise.Sq() == approx(0.73428125)

def test_Sp():
	assert surface.Sp() == approx(1.0)
	assert surface_noise.Sp() == approx(1.86741198)

def test_Sv():
	assert surface.Sv() == approx(1.0)
	assert surface_noise.Sv() == approx(1.92888076)

def test_Sz():
	assert surface.Sz() == approx(2.0)
	assert surface_noise.Sz() == approx(3.79629274)

def test_Ssk():
	assert surface.Ssk() == approx(0.0)
	assert surface_noise.Ssk() == approx(-0.05357154)

def test_Sku():
	assert surface.Sku() == approx(1.50000001)
	assert surface_noise.Sku() == approx(1.71930405)

def test_Sdr():
	assert surface.Sdr() == approx(130.08211683)
	assert surface_noise.Sdr() == approx(230.56169546)

def test_Sk():
	assert surface.Sk() == approx(1.725)
	assert surface_noise.Sk() == approx(2.22396096)

def test_Spk():
	assert surface.Spk() == approx(0.5236)
	assert surface_noise.Spk() == approx(0.18799396)

def test_Svk():
	assert surface.Svk() == approx(0.0)
	assert surface_noise.Svk() == approx(0.26373285)

def test_Smr1():
	assert surface.Smr1() == approx(25.0)
	assert surface_noise.Smr1() == approx(2.13617563)

def test_Smr2():
	assert surface.Smr2() == approx(95.0)
	assert surface_noise.Smr2() == approx(92.75723063)

def test_period():
	assert surface.period() == approx(2.0)
	assert surface_noise.period() == approx(8.33333333)

def test_homogeneity():
	assert surface.homogeneity() == approx(1.0)
	assert surface_noise.homogeneity() == approx(0.9598)

def test_depth():
	assert surface.depth() == approx(1.90211303)
	assert surface_noise.depth() == approx(1.86919941)

def test_aspect_ratio():
	assert surface.aspect_ratio() == approx(0.95105652)
	assert surface_noise.aspect_ratio() == approx(0.22430393)

