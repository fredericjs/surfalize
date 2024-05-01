============
Installation
============

To install `surfalize` from PyPI run:

.. code::

    pip install surfalize

Currently, wheels on PyPI are only for Windows, in the future the support for manylinux wheels is planned.
Alternatively, you can clone the git repository, navigate to the root folder and run:

.. code::

    pip install .


However, you will need to have both `Cython` and a C-Compiler installed (MSVC on Windows,
gcc on Linux, MinGW is not supported currently). If you install in editable mode using

.. code::

    pip install -e .


be aware that a change of the pyx files does not reinvoke the Cython build process.

You can also install directly from git via pip:

.. code::

    pip install git+https://github.com/fredericjs/surfalize.git

Optionally, you can specify the branch to install the latest updates from the develop branch:
.. code::
    pip install git+https://github.com/fredericjs/surfalize.git@develops
