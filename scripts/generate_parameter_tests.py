import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", action='store', dest='output', help="Directs the output to the specified file",
                    default='parameter_tests.py')
args = parser.parse_args()


preamble = """from surfalize import Surface
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
"""

exec(preamble)

text = preamble + '\n\n'
AVAILABLE_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sk', 'Spk', 'Svk', 'Smr1', 'Smr2', 'period', 'homogeneity', 'depth', 'aspect_ratio')
surf_objs = {'surface': surface, 'surface_noise': surface_noise}
for par in AVAILABLE_PARAMETERS:
    test = f'def test_{par}():\n'
    for name, surf_obj in surf_objs.items():
        res = round(getattr(surf_obj, par)(), 8)
        test += f'\tassert {name}.{par}() == approx({res})\n'
    text += test + '\n'

with open(args.output, 'w') as file:
    file.write(text)