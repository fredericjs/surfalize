import os
import io
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'
import tqdm
import click
from .surface import Surface
from .file.loader import supported_formats_read

def convert_file(input_path, output_path, format):
    surface = Surface.load(input_path)
    surface.save(output_path)

@click.group()
def cli():
    """A command-line tool for surfalize package."""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
def view(input_path):
    input_path = Path(input_path)
    Surface.load(input_path).plot_2d()
    plt.gcf().canvas.manager.set_window_title(input_path.name)
    plt.show()


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--format', '-f', default='default_format', help='Format to convert the file(s) to.')
def convert(input_path, output_path, format):
    """
    Convert a file or all files in a directory to another format.

    INPUT_PATH can be a file or a directory. If it is a directory, all files inside
    will be converted and saved to the OUTPUT_PATH directory.
    """

    if os.path.isfile(input_path):
        # Convert a single file
        convert_file(input_path, output_path, format)
        click.echo(f"Converted {input_path} to {output_path} in format {format}.")
    elif os.path.isdir(input_path):
        # Convert all files in a directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            convert_file(input_file, output_file, format)
            click.echo(f"Converted {input_file} to {output_file} in format {format}.")
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

    surface.plot_abbott_curve()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    #pdf.image(buf, x=10, y=20, w=100, h=0)  # Height is automatically scaled

    data = surface.roughness_parameters()

    # Set font for the table header
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(100, 10, "Key", border=1, align='C')
    pdf.cell(100, 10, "Value", border=1, align='C')
    pdf.ln()  # Move to the next line

    # Set font for the table rows
    pdf.set_font("Arial", size=12)

    # Iterate over the dictionary and add each key-value pair to the table
    for key, value in data.items():
        pdf.cell(100, 10, str(key), border=1)
        pdf.cell(100, 10, str(round(value, 3)), border=1)
        pdf.ln()  # Move to the next line

    # Output the PDF
    pdf.output("pdf_with_matplotlib_image.pdf")

    print("PDF report created successfully.")