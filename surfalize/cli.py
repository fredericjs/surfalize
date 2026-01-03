import functools
import io
from pathlib import Path
import subprocess
import platform

import click
import matplotlib
import matplotlib.pyplot as plt

from .surface import Surface
from .file import supported_formats_read, supported_formats_write


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

PARAMETER_UNITS = {
    'Sa': 'µm',
    'Sq': 'µm',
    'Sv': 'µm',
    'Sp': 'µm',
    'Sz': 'µm',
    'Ssk': '',
    'Sku': '',
    'Sdr': '%',
    'Sdq': '',
    'Sk': 'µm',
    'Spk': 'µm',
    'Svk': 'µm',
    'Sxp': 'µm',
    'Sal': 'µm',
    'Str': '',
    'Smr1': '%',
    'Smr2': '%',
    'Vmp': 'µm³/µm²',
    'Vmc': 'µm³/µm²',
    'Vvc': 'µm³/µm²',
    'Vvv': 'µm³/µm²',
    'Period': 'µm',
    'Depth': 'µm',
    'Aspect ratio': '',
    'Homogeneity': ''
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def open_pdf(file_path: Path):
    system = platform.system()
    if system == 'Darwin':
        subprocess.run(['open', str(file_path)], check=False)
    elif system == 'Windows':
        subprocess.run(['start', '', str(file_path)], shell=True, check=False)
    elif system == 'Linux':
        subprocess.run(['xdg-open', str(file_path)], check=False)
    else:
        raise NotImplementedError(f'Operating system {system} is not supported.')


def convert_file(input_path, output_path, skip_image_layers, **kwargs):
    surface = Surface.load(input_path, read_image_layers=not skip_image_layers)
    surface.save(output_path, **kwargs)


def perform_surface_operations(surface, **kwargs):
    if kwargs['fill_nonmeasured']:
        surface.fill_nonmeasured(inplace=True)

    if kwargs['level']:
        surface.level(inplace=True)

    if kwargs['center']:
        surface.center(inplace=True)

    if kwargs['zero']:
        surface.zero(inplace=True)

    if kwargs['threshold'] is not None:
        surface.threshold(kwargs['threshold'], inplace=True)
        surface.fill_nonmeasured(inplace=True)

    if kwargs['remove_outliers'] is not None:
        surface.remove_outliers(n=kwargs['remove_outliers'], inplace=True)
        surface.fill_nonmeasured(inplace=True)

    if kwargs['highpass'] is not None:
        surface.filter('highpass', cutoff=kwargs['highpass'], inplace=True)

    if kwargs['lowpass'] is not None:
        surface.filter('lowpass', cutoff=kwargs['lowpass'], inplace=True)

    if kwargs['bandpass'] is not None:
        surface.filter(
            'bandpass',
            cutoff=kwargs['bandpass'][0],
            cutoff2=kwargs['bandpass'][1],
            inplace=True
        )


# ---------------------------------------------------------------------
# Click option bundle
# ---------------------------------------------------------------------

def common_options(function):
    @click.option('--level', '-l', is_flag=True, help='Level the topography')
    @click.option('--fill-nonmeasured', '-fn', is_flag=True,
                  help='Fill the non-measured points of the topography')
    @click.option('--center', '-c', is_flag=True, help='Center the topography')
    @click.option('--zero', '-z', is_flag=True, help='Zero the topography')
    @click.option('--highpass', '-hp', type=float, help='Highpass filter frequency')
    @click.option('--lowpass', '-lp', type=float, help='Lowpass filter frequency')
    @click.option('--bandpass', '-bp', nargs=2, type=float,
                  help='Bandpass filter frequencies (low high)')
    @click.option('--threshold', '-t', nargs=2, type=float, default=None,
                  help='Threshold surface based on material ratio curve')
    @click.option('--remove-outliers', '-ro', type=int,
                  help='Remove outliers above N sigma')
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if sum(v is not None for v in
               (kwargs['highpass'], kwargs['lowpass'], kwargs['bandpass'])) > 1:
            raise click.BadParameter(
                'Only one of --highpass, --lowpass, or --bandpass may be used.'
            )

        if kwargs['zero'] and kwargs['center']:
            raise click.BadParameter(
                'Only one of --zero or --center may be used.'
            )

        return function(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """A command-line tool for surfalize."""
    pass


# ---------------------------------------------------------------------
# SHOW
# ---------------------------------------------------------------------

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--plot-3d', is_flag=True, help='Plot the surface in 3D')
@common_options
def show(input_path, plot_3d, **kwargs):
    """
    Show a plot of the surface in 2D or 3D.
    """
    input_path = Path(input_path)
    surface = Surface.load(input_path)
    perform_surface_operations(surface, **kwargs)

    if plot_3d:
        from surfalize.plotting import plot_3d
        plot_3d(surface, interactive=True, window_title=input_path.name)
        return

    # 2D plotting
    if matplotlib.get_backend().lower() == 'agg':
        click.echo(
            "No interactive matplotlib backend found.\n"
            "Install PySide6/PyQt5 or python3-tk, or use --plot-3d."
        )
        return

    import matplotlib.pyplot as plt

    with plt.rc_context({'toolbar': 'None'}):
        surface.plot_2d()
        plt.gcf().canvas.manager.set_window_title(input_path.name)
        plt.show()


# ---------------------------------------------------------------------
# CONVERT
# ---------------------------------------------------------------------

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--format', '-f', default='.sur',
              help='Format to convert the file(s) to.')
@click.option('--skip-image-layers', is_flag=True,
              help='Only convert topography layer and skip image layers.')
@click.option('--compressed', is_flag=True,
              help='Use compressed format if available.')
def convert(input_path, output_path, format, skip_image_layers, compressed):
    """
    Convert a file or all files in a directory to another format.
    """
    if output_path.is_file():
        format = output_path.suffix

    if format != '.sur' and compressed:
        raise click.BadParameter(
            "The --compressed flag is only valid for .sur format."
        )

    if format not in supported_formats_write:
        click.echo(
            f'Output format {format} not supported.\n'
            f'Available formats: {", ".join(supported_formats_write)}'
        )
        return

    kwargs = {}
    if format == '.sur':
        kwargs['compressed'] = compressed

    if input_path.is_file():
        convert_file(input_path, output_path, skip_image_layers, **kwargs)
        click.echo(f"Converted {input_path} → {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for input_file in input_path.iterdir():
        if input_file.is_file() and input_file.suffix in supported_formats_read:
            out = output_path / (input_file.stem + format)
            convert_file(input_file, out, skip_image_layers, **kwargs)
            click.echo(f"Converted {input_file} → {out}")


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--open-after', is_flag=True)
@click.option('--periodic-parameters', '-pp', is_flag=True)
@common_options
def report(input_path, open_after, periodic_parameters, **kwargs):
    """
    Generate a PDF report for a surface.
    """
    from fpdf import FPDF
    from fpdf.fonts import FontFace
    import matplotlib.pyplot as plt

    input_path = Path(input_path).absolute()

    if input_path.is_dir():
        files = []
        for ext in supported_formats_read:
            files.extend(input_path.glob(f'*{ext}'))
    else:
        files = [input_path]

    for file in files:
        surface = Surface.load(file)
        perform_surface_operations(surface, **kwargs)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_right_margin(20)
        pdf.set_font("Arial", size=12)
        pdf.text(20, 20, f'REPORT: {file.name}')

        # 2D plot
        surface.plot_2d()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        pdf.image(buf, x=20, y=30, w=80)

        # 3D plot
        buf = io.BytesIO()
        surface.plot_3d().save(buf, format='PNG')
        buf.seek(0)
        pdf.image(buf, x=105, y=20, w=90)

        # Parameters
        data = surface.roughness_parameters()

        if periodic_parameters:
            periodic_data = {
                'Period': surface.period(),
                'Depth': surface.depth()[0],
                'Aspect ratio': surface.aspect_ratio(),
                'Homogeneity': surface.homogeneity()
            }

        grey = (128, 128, 128)
        lightgrey = (200, 200, 200)
        headings = FontFace(emphasis="BOLD", fill_color=grey)
        sub = FontFace(emphasis="BOLD", fill_color=lightgrey)

        pdf.set_y(90)
        with pdf.table(width=60, line_height=1.5 * pdf.font_size,
                       headings_style=headings) as table:

            row = table.row()
            row.cell('ISO 25178', colspan=3)

            def group(title, params):
                row = table.row()
                row.cell(title, colspan=3, style=sub)
                for p in params:
                    row = table.row()
                    row.cell(p)
                    row.cell(f'{data[p]:.3f}')
                    row.cell(PARAMETER_UNITS[p])

            group('Spatial', ['Sa', 'Sq', 'Sv', 'Sp', 'Sz'])
            group('Functional', ['Sk', 'Spk', 'Svk'])

            if periodic_parameters:
                row = table.row()
                row.cell('Periodic', colspan=3, style=sub)
                for p, v in periodic_data.items():
                    row = table.row()
                    row.cell(p)
                    row.cell(f'{v:.3f}')
                    row.cell(PARAMETER_UNITS[p])

        output = file.with_suffix(file.suffix + '.pdf')
        pdf.output(output)
        click.echo(f"PDF report created: {output}")

        if open_after:
            open_pdf(output)