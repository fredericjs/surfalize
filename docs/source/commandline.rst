====================
Command line scripts
====================

Surfalize offers a variety of scripts to perform a number of operations directly from the command line. For this
purpose, an entry point is defined that can be called by invoking `surfalize` in the command line.

Surfaces can be viewed using:

.. code::

    surfalize view example_file.vk4

Files of any supported file format can be convert to a any fileformat support for writing by invoking:

.. code::

    surfalize convert path/to/example_file.vk4 path/to/converted_example_file.sur

To convert all files from a folder to another format, the paths should be folders and the `--format` option is used to
specify the target format:

.. code::

    surfalize convert path/to/files path/to/convert_files --format .sur

Optionally, keyword arguments used during saving can be specified with additional flags, such an the `--compressed` flag
for the .sur file format:

.. code::

    surfalize convert path/to/files path/to/convert_files --format .sur

