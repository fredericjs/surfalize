from setuptools import setup, find_packages
import glob

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

def get_version():
    with open('surfalize/_version.py', 'r') as file:
        content = file.read()
    return content.split()[-1].strip("'")

setup(
    name='surfalize',
    version=get_version(),
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
        'python-dateutil',
        'pillow'
    ],
)
