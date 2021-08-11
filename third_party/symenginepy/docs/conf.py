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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import symengine

# -- Project information -----------------------------------------------------

project = 'symengine'
copyright = '2021, SymEngine development team <symengine@googlegroups.com>'
author = 'SymEngine development team <symengine@googlegroups.com>'

# The full version, including alpha/beta/rc tags
release = symengine.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Consumes docstrings
    "sphinx.ext.napoleon",  # Allows for Google Style Docs
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.intersphinx",  # Connects to other documentation
    "sphinx.ext.todo",  # Show TODO details
    "sphinx.ext.imgconverter",  # Handle svg images
    "sphinx.ext.duration",  # Shows times in the processing pipeline
    "sphinx.ext.mathjax",  # Need math support
    "sphinx.ext.githubpages",  # Puts the .nojekyll and CNAME files
    "sphinxcontrib.apidoc",  # Automatically sets up sphinx-apidoc
    # "recommonmark", # Parses markdown
    "m2r2", # Parses markdown in rst
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# API Doc settings
apidoc_module_dir = "../"
apidoc_output_dir = "source"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "Symengine Python Bindings"
# html_logo = "path/to/logo.png"
# html_favicon = "path/to/favicon.ico"
html_theme_options = {
    "repository_url": "https://github.com/symengine/symengine.py",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "use_download_button": True,
    "home_page_in_toc": True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
