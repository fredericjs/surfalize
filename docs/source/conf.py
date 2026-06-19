# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx.ext.autodoc import ClassDocumenter
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'surfalize'
copyright = '2023, Frederic Schell'
author = 'Frederic Schell'
#release = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

numpydoc_class_members_toctree = False

# Render the stored notebook outputs as-is; do not re-execute notebooks at build
# time, since the examples rely on local measurement data that isn't available
# in the docs environment.
nbsphinx_execute = 'never'

autodoc_mock_imports = [
    #'numpy',
    'matplotlib',
    'pandas',
    'scipy',
    'tqdm',
    'sklearn',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
html_static_path = ['_static']
html_css_files = ['custom.css']