import io
from pathlib import Path
import subprocess
import platform
import matplotlib.pyplot as plt
import click

from .surface import Surface
from .file.loader import supported_formats_read, supported_formats_write

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

def open_pdf(file_path):
    system = platform.system()
    if system == 'Darwin':  # macOS
        subprocess.call(['open', file_path])
    elif system == 'Windows':  # Windows
        subprocess.call(['start', '', file_path], shell=True)
    elif system == 'Linux':  # Linux
        subprocess.call(['xdg-open', file_path])
    else:
        raise NotImplementedError(f'Operating system {system} is not supported.')

def convert_file(input_path, output_path, skip_image_layers, *args, **kwargs):
    surface = Surface.load(input_path, read_image_layers=not skip_image_layers)
    surface.save(output_path, *args, **kwargs)

@click.group()
def cli():
    """A command-line tool for surfalize package."""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--level', '-l', is_flag=True, help='Level the surface topography.')
def show(input_path, level=False):
    plt.rcParams['toolbar'] = 'None'
    input_path = Path(input_path)
    surface = Surface.load(input_path)
    if level:
        surface.level(inplace=True)
    surface.plot_2d()
    plt.gcf().canvas.manager.set_window_title(input_path.name)
    plt.show()

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--format', '-f', default='.sur', help='Format to convert the file(s) to.')
@click.option('--skip-image-layers', 'skip_image_layers', is_flag=True, help='Only conver topography layer and skip image layers.')
@click.option('--compressed', is_flag=True, help='Use the compressed version of the format if available.')
def convert(input_path: Path, output_path: Path, format: str, skip_image_layers: bool, compressed: bool):
    """
    Convert a file or all files in a directory to another format.

    INPUT_PATH can be a file or a directory. If it is a directory, all files inside
    will be converted and saved to the OUTPUT_PATH directory.
    """
    if output_path.is_file():
        format = output_path.suffix
    if format != '.sur' and compressed:
        raise click.BadParameter("The --compressed flag is only valid when format is .sur")
    if format not in supported_formats_write:
        click.echo(f'Output format {format} not supported.\n'
                   f'Available formats are: {", ".join(supported_formats_write)}')
        return
    kwargs = dict()
    if format == '.sur':
        kwargs['compressed'] = compressed
    if input_path.is_file():
        # Convert a single file
        convert_file(input_path, output_path,skip_image_layers, **kwargs)
        click.echo(f"Converted {input_path} to {output_path} in format {format}.")
    elif input_path.is_dir():
        # Convert all files in a directory
        output_path.mkdir(parents=True, exist_ok=True)
        for input_file in input_path.iterdir():
            if input_file.is_file() and input_file.suffix in supported_formats_read:
                output_file = output_path / (input_file.stem + format)

                convert_file(input_file, output_file, skip_image_layers, **kwargs)
                click.echo(f"Converted {input_file} to {output_file}.")
    else:
        click.echo("Input path must be a file or directory.")

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--open-after', 'open_after', is_flag=True)
@click.option('--level', '-l', is_flag=True, help='Level the topography')
@click.option('--center', '-c', is_flag=True, help='Center the topography')
@click.option('--zero', '-z', is_flag=True, help='Zero the topography')
@click.option('--highpass', type=float, help='Highpass filter frequency.')
@click.option('--lowpass', type=float, help='Lowpass filter frequency.')
@click.option('--bandpass', nargs=2, type=float, help='Bandpass filter frequencies (low, high).')
def report(input_path, open_after, level, center, zero, highpass, lowpass, bandpass):
    if sum([v is not None for v in [highpass, lowpass, bandpass]]) > 1:
        raise click.BadParameter('Only one of the options --highpass, --lowpass, or --bandpass can be used at a time.')
    if sum([v for v in [zero, center]]) > 1:
        raise click.BadParameter('Only one of the options --zero, --center can be used at a time.')

    input_path = Path(input_path)
    from fpdf import FPDF
    from fpdf.fonts import FontFace
    surface = Surface.load(input_path).fill_nonmeasured()
    if level:
        surface.level(inplace=True)
    if center:
        surface.center(inplace=True)
    if zero:
        surface.zero(inplace=True)
    if highpass is not None:
        surface.filter('highpass', cutoff=highpass, inplace=True)
    if lowpass is not None:
        surface.filter('lowpass', cutoff=lowpass, inplace=True)
    if bandpass is not None:
        surface.filter('bandpass', cutoff=bandpass[0], cutoff2=bandpass[1], inplace=True)
    ax = surface.plot_2d()

    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # Create an FPDF instance
    pdf = FPDF()
    pdf.add_page()
    pdf.set_right_margin(20)
    pdf.set_font("Arial", size=12)
    pdf.text(20, 20, f'REPORT: {input_path.name}')

    # Insert the Matplotlib image into the PDF
    pdf.image(buf, x=20, y=30, w=100, h=0)  # Height is automatically scaled

    padding = 10
    pdf.set_y(pdf.get_y() + padding)

    profile = surface.get_horizontal_profile(surface.height_um/2, average=10)
    profile.plot_2d()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    pdf.image(buf, x=24, y=100, w=84, h=0)

    surface.plot_abbott_curve()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    pdf.image(buf, x=25, y=130, w=80, h=0)  # Height is automatically scaled

    pdf.set_font("Arial", size=8)
    grey = (128, 128, 128)
    lightgrey = (200, 200, 200)
    headings_style = FontFace(emphasis="BOLD", fill_color=grey)
    subheadings_style = FontFace(emphasis="BOLD", fill_color=lightgrey)

    data = surface.roughness_parameters()
    periodic_data = {
        'Period': surface.period(),
        'Depth': surface.depth()[0],
        'Aspect ratio': surface.aspect_ratio(),
        'Homogeneity': surface.homogeneity()
    }

    def create_group(table, params, title):
        row = table.row()
        row.cell(title, colspan=3, style=subheadings_style)
        for param in params:
            row = table.row()
            row.cell(param)
            row.cell(f'{data[param]:.3f}')
            row.cell(PARAMETER_UNITS[param])

    pdf.set_y(30)
    with pdf.table(width=60, line_height=1.5 * pdf.font_size, text_align="LEFT", align='RIGHT', headings_style=headings_style) as table:
        row = table.row()
        row.cell('ISO 25178', colspan=3)

        create_group(table, ['Sa', 'Sq', 'Sv', 'Sp', 'Sz', 'Ssk', 'Sku'], 'Spatial parameters')
        create_group(table, ['Sdr', 'Sdq'], 'Hybrid parameters')
        create_group(table, ['Sal', 'Str'], 'Height parameters')
        create_group(table, ['Sk', 'Spk', 'Svk', 'Smr1', 'Smr2', 'Sxp'], 'Functional parameters')
        create_group(table, ['Vmc', 'Vmp', 'Vvc', 'Vvv'], 'Functional parameters (Volume')

        row = table.row()
        row.cell('Periodic parameters', colspan=3, style=subheadings_style)
        for param in periodic_data.keys():
            row = table.row()
            row.cell(param)
            row.cell(f'{periodic_data[param]:.3f}')
            row.cell(PARAMETER_UNITS[param])

    # Output the PDF
    output_file = Path(input_path).with_suffix('.pdf')
    pdf.output(output_file)

    print("PDF report created successfully.")

    if open_after:
        open_pdf(output_file)