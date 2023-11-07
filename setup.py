from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


# Define your Cython extension module
extensions = [
    Extension(
        'surfalize.calculations',  # Replace with the path to your Cython module
        sources=['surfalize/calculations.pyx'],  # Replace with the path to your .pyx file
    ),
]

# Use cythonize to compile the extension modules
ext_modules = cythonize(extensions)

# Define your package's metadata
setup(
    name='surfalize',
    version='0.0.1',
    description='A python module to analyze surface roughness',
    author='Frederic Schell',
    author_email='frederic.schell@iws.fraunhofer.de',
    packages=find_packages(),
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
        'numpy>=1.18.1',
        'matplotlib>=3.1.1',
        'pandas>=1.0.1',
        'scipy>=1.4.1',
    ]
    ext_modules=(ext_modules,)  # Include the Cython extension modules
)
