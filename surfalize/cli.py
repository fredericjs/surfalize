import io
from pathlib import Path
import matplotlib.pyplot as plt
import click
from .surface import Surface
from .file.loader import supported_formats_read, supported_formats_write

def convert_file(input_path, output_path, *args, **kwargs):
    surface = Surface.load(input_path, read_image_layers=True)
    surface.save(output_path, *args, **kwargs)

@click.group()
def cli():
    """A command-line tool for surfalize package."""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
def view(input_path):
    plt.rcParams['toolbar'] = 'None'
    input_path = Path(input_path)
    Surface.load(input_path).plot_2d()
    plt.gcf().canvas.manager.set_window_title(input_path.name)
    plt.show()

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--format', '-f', default='.sur', help='Format to convert the file(s) to.')
@click.option('--compressed', is_flag=True, help='A special flag only used with the specific_format')
def convert(input_path: Path, output_path: Path, format: str, compressed: bool):
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
                   f'Available formats are: {', '.join(supported_formats_write)}')
        return
    if input_path.is_file():
        # Convert a single file
        convert_file(input_path, output_path, format)
        click.echo(f"Converted {input_path} to {output_path} in format {format}.")
    elif input_path.is_dir():
        # Convert all files in a directory
        output_path.mkdir(parents=True, exist_ok=True)
        for input_file in input_path.iterdir():
            if input_file.is_file() and input_file.suffix in supported_formats_read:
                output_file = output_path / (input_file.stem + format)
                kwargs = dict()
                if format == '.sur':
                    kwargs['compressed'] = compressed
                convert_file(input_file, output_file, **kwargs)
                click.echo(f"Converted {input_file} to {output_file}.")
    else:
        click.echo("Input path must be a file or directory.")

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
def report(input_path):
    from fpdf import FPDF
    surface = Surface.load(input_path).fill_nonmeasured()
    ax = surface.plot_2d()

    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # Create an FPDF instance
    pdf = FPDF()

    # Add a page to the PDF
    pdf.add_page()

    # Insert the Matplotlib image into the PDF
    pdf.image(buf, x=10, y=10, w=100, h=0)  # Height is automatically scaled

    padding = 10
    pdf.set_y(pdf.get_y() + padding)

    surface.plot_abbott_curve()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    pdf.image(buf, x=10, y=20, w=100, h=0)  # Height is automatically scaled

    # data = surface.roughness_parameters()
    #
    # # Set font for the table header
    # pdf.set_font("Arial", style='B', size=12)
    # pdf.cell(100, 10, "Key", border=1, align='C')
    # pdf.cell(100, 10, "Value", border=1, align='C')
    # pdf.ln()  # Move to the next line
    #
    # # Set font for the table rows
    # pdf.set_font("Arial", size=12)
    #
    # # Iterate over the dictionary and add each key-value pair to the table
    # for key, value in data.items():
    #     pdf.cell(100, 10, str(key), border=1)
    #     pdf.cell(100, 10, str(round(value, 3)), border=1)
    #     pdf.ln()  # Move to the next line

    # Output the PDF
    pdf.output("pdf_with_matplotlib_image.pdf")

    print("PDF report created successfully.")