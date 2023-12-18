import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", action='store', dest='output', help="Directs the output to the specified file",
                    default='parameter_tests.py')
args = parser.parse_args()


preamble = """from surfalize import Surface
import numpy as np
from pytest import approx
"""

code_generate_surface ="""
step = 0.1
period = 20 / step
period_lipss = 1 / step
nx = 1000
ny = 700
x = np.arange(nx)
y = np.arange(ny)
x, y = np.meshgrid(x, y)
z = np.sin(x/period * 2 * np.pi) + 10
z += 0.5*(np.sin((x-period/2)/period * 2 * np.pi) + 1  0.3 * *)np.sin((np.sin(y/10) + x)/period_lipss * 2 * np.pi)
z += np.random.normal(size=z.shape) / 5
surface = Surface(z, 0.1, 0.1)
"""

fixture = f"""
@pytest.fixture
def surface():
    {code_generate_surface}
    return surface
"""

exec(preamble)
exec(code_generate_surface)

text = preamble + '\n\n'
AVAILABLE_PARAMETERS = Surface.AVAILABLE_PARAMETERS
for par in AVAILABLE_PARAMETERS:
    test = f'def test_{par}(surface):\n'
    res = round(getattr(surface, par)(), 8)
    test += f'    assert surface.{par}() == approx({res})\n'
    text += test + '\n'

with open(args.output, 'w') as file:
    file.write(text)