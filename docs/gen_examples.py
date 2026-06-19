"""One-off generator for the example notebooks under source/examples/.

Builds each notebook from the cell definitions below, executes it so the
outputs (images, values) are embedded, and writes the .ipynb. Notebooks load
data through ``surfalize.examples`` so they are portable and need no local
files. Re-run from the docs/ directory: ``python gen_examples.py``.
"""
import os
import tempfile
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

OUT_DIR = os.path.join(os.path.dirname(__file__), 'source', 'examples')

# Each notebook is a list of (kind, source) tuples. kind is 'md' or 'code'.
NOTEBOOKS = {
    'file_io': [
        ('md', "# Loading and saving files\n\n"
               "surfalize reads a wide range of topography formats "
               "(`.vk4/6/7`, `.sur`, `.sdf`, `.plu`, `.plux`, `.al3d`, `.opd`, "
               "`.x3p`, `.gwy`, `.nms`, `.zmg`, `.os3d`, `.tmd`, `.xyz`, ...) "
               "through a single `Surface.load` entry point, and can write "
               "several of them back out. This example uses the bundled example "
               "files, which are downloaded on demand from the surfalize "
               "repository."),
        ('md', "## Discovering example files\n\n"
               "`list_examples` returns the measurement files shipped with the "
               "project. Pass a suffix to filter by format."),
        ('code', "from surfalize.examples import list_examples\n\n"
                 "list_examples('.vk4')"),
        ('md', "## Loading a measurement\n\n"
               "Every example exposes a `.load()` method returning a `Surface`. "
               "Stating the surface as the last line of a cell renders its "
               "topography directly."),
        ('code', "surface = list_examples('.vk4')[0].load()\n"
                 "surface"),
        ('md', "The `Surface` carries its physical dimensions, so analysis "
               "results come out in real units (µm)."),
        ('code', "print(surface)\n"
                 "print('Pixels:        ', tuple(surface.size))\n"
                 "print('Width x Height:', f'{surface.width_um:.1f} x "
                 "{surface.height_um:.1f} µm')"),
        ('md', "## Saving and round-tripping\n\n"
               "`Surface.save` writes the topography to another format — here "
               "the SURF (`.sur`) format. Reloading it yields an equivalent "
               "surface."),
        ('code', "surface.save('measurement.sur')\n\n"
                 "from surfalize import Surface\n"
                 "Surface.load('measurement.sur')"),
    ],
    'profiles': [
        ('md', "# Extracting and analyzing profiles\n\n"
               "Besides areal (3D) parameters, surfalize can extract 2D "
               "profiles from a surface and compute profile roughness "
               "parameters from them."),
        ('code', "from surfalize import Surface\n"
                 "from surfalize.examples import list_examples\n\n"
                 "surface = list_examples('.vk4')[0].load().level()\n"
                 "surface"),
        ('md', "## Horizontal, vertical and oblique profiles\n\n"
               "Horizontal and vertical profiles are taken at a coordinate "
               "given in µm. An oblique profile runs between two points. Each "
               "returns a `Profile` object, which renders as a line plot."),
        ('code', "profile = surface.get_horizontal_profile(35)  # y = 35 µm\n"
                 "profile"),
        ('code', "surface.get_vertical_profile(47)  # x = 47 µm"),
        ('code', "surface.get_oblique_profile(0, 0, 90, 70)"),
        ('md', "## Profile roughness parameters\n\n"
               "`Profile` exposes the standard profile parameters (Ra, Rq, Rz, "
               "Rsk, Rku) as well as the dominant spatial period."),
        ('code', "print(f'Ra     = {profile.Ra():.3f} µm')\n"
                 "print(f'Rq     = {profile.Rq():.3f} µm')\n"
                 "print(f'Rz     = {profile.Rz():.3f} µm')\n"
                 "print(f'Period = {profile.period():.3f} µm')"),
        ('md', "## Plotting a profile\n\n"
               "`plot_2d` draws the extracted profile on a matplotlib axis."),
        ('code', "profile.plot_2d();"),
    ],
    'texture_analysis': [
        ('md', "# Analyzing periodic surface textures\n\n"
               "surfalize is built primarily for periodically microtextured "
               "surfaces. This example shows the parameters and visualizations "
               "tailored to that use case, using a laser-textured sample with a "
               "near-1D periodic structure."),
        ('code', "from surfalize import Surface\n"
                 "from surfalize.examples import list_examples\n\n"
                 "surface = list_examples('.vk4')[0].load().level()\n"
                 "surface"),
        ('md', "## Spatial period and orientation\n\n"
               "`period` returns the dominant spatial period, `orientation` the "
               "angle of the texture, and `aspect_ratio` the ratio of the two "
               "principal periods."),
        ('code', "print(f'Period:       {surface.period():.3f} µm')\n"
                 "print(f'Orientation:  {surface.orientation():.2f}°')\n"
                 "print(f'Aspect ratio: {surface.aspect_ratio():.3f}')"),
        ('md', "## Structure depth\n\n"
               "`depth` estimates the mean peak-to-valley depth of the periodic "
               "structure (and its standard deviation) by sampling many "
               "profiles across the surface."),
        ('code', "depth_mean, depth_std = surface.depth()\n"
                 "print(f'Depth: {depth_mean:.3f} ± {depth_std:.3f} µm')"),
        ('md', "## Homogeneity\n\n"
               "`homogeneity` returns a value between 0 and 1 quantifying how "
               "uniform the texture is across the field of view."),
        ('code', "surface.homogeneity()"),
        ('md', "## Autocorrelation function\n\n"
               "The areal autocorrelation function reveals the periodicity and "
               "directionality of the texture."),
        ('code', "surface.plot_autocorrelation();"),
        ('md', "## Fourier transform\n\n"
               "The 2D power spectrum shows the spatial frequencies that make "
               "up the texture."),
        ('code', "surface.plot_fourier_transform(hanning=True);"),
        ('md', "## Angular power spectrum\n\n"
               "Integrating the power spectrum over radius as a function of "
               "angle highlights the dominant texture orientation."),
        ('code', "surface.plot_angular_power_spectrum();"),
    ],
}


def build(cells):
    nb = new_notebook()
    for kind, src in cells:
        nb.cells.append(
            new_markdown_cell(src) if kind == 'md' else new_code_cell(src)
        )
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'
    }
    return nb


def main():
    workdir = tempfile.mkdtemp()  # execute here so saved files don't pollute repo
    for name, cells in NOTEBOOKS.items():
        nb = build(cells)
        print(f'Executing {name} ...', flush=True)
        NotebookClient(nb, timeout=300, resources={'metadata': {'path': workdir}}).execute()
        path = os.path.join(OUT_DIR, f'{name}.ipynb')
        nbformat.write(nb, path)
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
