# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

if not os.path.exists("_build_temp"):
    import shutil
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        shutil.copytree("../..", Path(d).joinpath("_build_temp"))
        shutil.copytree(Path(d).joinpath("_build_temp"), "_build_temp")

from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('_build_temp/python'))
sys.path.insert(0, os.path.abspath('_build_temp/python/fate_client'))
sys.path.insert(0, os.path.abspath('_build_temp/python/fate_test'))

# -- Project information -----------------------------------------------------

project = 'FATE'
copyright = '2020, FederatedAI'
author = 'FederatedAI'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autodocsumm',
    'recommonmark',
    'sphinx_click.ext',
    'sphinx_markdown_tables'
]

autosummary_generate = True

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
    ],
}
add_module_names = False
master_doc = 'index'


# hack to replace rst file link to html link
def ultimateReplace(app, docname, source):
    result = source[0]
    result = result.replace(".rst", ".html")
    result = result.replace(".md", ".html")
    source[0] = result


def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', ultimateReplace)
