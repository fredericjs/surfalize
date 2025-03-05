====================
Command line scripts
====================

Surfalize offers a variety of scripts to perform a number of operations directly from the command line. For this
purpose, an entry point is defined that can be called by invoking :code:`surfalize` in the command line. To use the command
line interface, install `surfalize` with optional dependencies :code:`[cli]` or :code:`[all]`:

.. code::

    pip install surfalize[cli]

Surfaces can be viewed using:

.. code::

    surfalize show example_file.vk4

Files of any supported file format can be convert to a any fileformat support for writing by invoking:

.. code::

    surfalize convert path/to/example_file.vk4 path/to/converted_example_file.sur

To convert all files from a folder to another format, the paths should be folders and the :code:`--format` option is
used to specify the target format:

.. code::

    surfalize convert path/to/files path/to/convert_files --format .sur

Optionally, keyword arguments used during saving can be specified with additional flags, such an the
:code:`--compressed` flag for the .sur file format:

.. code::

    surfalize convert path/to/files path/to/convert_files --format .sur


show
----

Show a plot of the surface in 2D or 3D.

.. code-block:: bash

    surfalize show [OPTIONS] INPUT_PATH

Options:
    --3d                      Plot the surface in 3d.
    -l, --level              Level the topography
    -fn, --fill-nonmeasured  Fill the non-measured points
    -c, --center             Center the topography
    -z, --zero               Zero the topography
    -hp, --highpass FLOAT    Highpass filter frequency
    -lp, --lowpass FLOAT     Lowpass filter frequency
    -bp, --bandpass FLOAT    Bandpass filter frequencies (low, high)
    -t, --threshold FLOAT    Threshold surface based on material ratio curve
    -ro, --remove-outliers INTEGER  Remove outliers

convert
-------

Convert a file or all files in a directory to another format.

.. code-block:: bash

    surfalize convert [OPTIONS] INPUT_PATH OUTPUT_PATH

Options:
    -f, --format TEXT        Format to convert the file(s) to (default: .sur)
    --skip-image-layers      Only convert topography layer and skip image layers
    --compressed            Use the compressed version of the format if available

The INPUT_PATH can be either a single file or a directory. When a directory is provided,
all supported files in that directory will be converted and saved to the OUTPUT_PATH directory.

report
------

Generate a PDF report for a surface.

.. code-block:: bash

    surfalize report [OPTIONS] INPUT_PATH

Options:
    --open-after            Open the PDF after generation
    --periodic-parameters   Include periodic parameter analysis
    -l, --level            Level the topography
    -fn, --fill-nonmeasured Fill the non-measured points
    -c, --center           Center the topography
    -z, --zero             Zero the topography
    -hp, --highpass FLOAT  Highpass filter frequency
    -lp, --lowpass FLOAT   Lowpass filter frequency
    -bp, --bandpass FLOAT  Bandpass filter frequencies (low, high)
    -t, --threshold FLOAT  Threshold surface based on material ratio curve
    -ro, --remove-outliers INTEGER  Remove outliers

The report includes:

* 2D and 3D surface plots
* Autocorrelation analysis
* Abbott curve
* ISO 25178 parameters
* Summary of applied operations

