from pathlib import Path
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

NUMPY_INCLUDE_DIR = numpy.get_include()
NUMPY_MACROS = ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

def is_package_dir(dirpath):
    """
    Returns True if directory contains an __init__.py file and is therefore a python package.
    """
    return '__init__.py' in [path.name for path in dirpath.iterdir()]

def resolve_full_module_name(modulepath):
    """
    Returns the full package of a module file, e.g. 'surfalize.roughness.height'.
    """
    package = [modulepath.stem]
    directory = modulepath.parent
    while is_package_dir(directory):
        package.append(directory.name)
        directory = directory.parent
    return '.'.join(reversed(package))

def compile_cython_extensions(root='.'):
    """
    Finds and compiles all cython extensions.
    """
    root = Path(root)
    extensions = []
    for pyxpath in root.rglob('*.pyx'):
        extension = Extension(resolve_full_module_name(pyxpath), [str(pyxpath)],
                              include_dirs=[NUMPY_INCLUDE_DIR],
                              define_macros=[NUMPY_MACROS])
        extensions.append(extension)
    ext_modules = cythonize(extensions)
    return ext_modules

ext_modules = compile_cython_extensions()

setup(
    name='surfalize',
    version='0.7.0',
    description='A python module to analyze surface roughness',
    author='Frederic Schell',
    author_email='frederic.schell@iws.fraunhofer.de',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'numpy>=1.18.1, <2.0',
        'matplotlib>=3.1.1',
        'pandas>=1.0.1',
        'scipy>=1.4.1',
        'tqdm>=4.64.1',
        'openpyxl>=3.1.2',
        'scikit-learn',
        'python-dateutil'
    ],
    ext_modules=ext_modules
)
