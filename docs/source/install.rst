============
Installation
============

To install `surfalize` from PyPI run `pip install surfalize`. Alternatively,
you can clone the git repository, navigate to the root folder and run

    ``
    pip install .
    ``

However, you will need to have both `Cython` and a C-Compiler installed (MSVC on Windows,
gcc on Linux, MinGW is not supported currently). If you install in editable mode using

    ``
    pip install -e .
    ``

be aware that a change of the pyx files does not reinvoke the Cython build process.