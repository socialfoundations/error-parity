# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'error-parity'
copyright = '2023, AndreFCruz'
author = 'AndreFCruz'

# Import package version programmatically
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from error_parity._version import __version__
release = __version__
version = __version__

# Copy examples folder to the documentation folder
import shutil
shutil.copytree(src="../examples", dst="examples", dirs_exist_ok=True)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',  # needs to be AFTER napoleon
    'numpydoc',
    'sphinx_copybutton',
    # 'sphinx_autopackagesummary',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',             # for rendering jupyter notebooks
    'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting',   # current work-around for syntax-highlighting on jupyter notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'
autosummary_generate = True
autodoc_typehints = 'description'
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

autosectionlabel_prefix_document = True

myst_heading_anchors = 3

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = [
    'custom.js',    # custom JS file
]

# nbsphinx configuration
nbsphinx_execute = 'never'  # Set to 'always' if you want to execute the notebooks during the build process

# numpydoc configuration
numpydoc_show_class_members = False
