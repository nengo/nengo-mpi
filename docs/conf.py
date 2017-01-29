# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import sys
import os

try:
    import sphinx_rtd_theme
except ImportError:
    print("To build the documentation the sphinx_rtd_theme "
          "must be installed in the current environment. Please install these "
          "and their requirements first. A virtualenv is recommended!")
    sys.exit(1)


base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

about = {}
with open(os.path.join(base_dir, "nengo_mpi", "__about__.py")) as f:
    exec(f.read(), about)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'numpydoc',
]

# -- sphinx.ext.todo config
todo_include_todos = True
# -- numpydoc config
numpydoc_show_class_members = False

# -- sphinx config
exclude_patterns = ['_build']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

# Need to include https Mathjax path for sphinx < v1.3
mathjax_path = ("https://cdn.mathjax.org/mathjax/latest/MathJax.js"
                "?config=TeX-AMS-MML_HTMLorMML")

project = u'nengo_mpi'
authors = about["__author__"]
copyright = about["__copyright__"]
version = '.'.join(about["__version__"].split('.')[:2])  # Short X.Y version
release = about["__version__"]  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "nengo_mpi {0} docs".format(release)
# html_static_path = ['_static']
html_use_smartypants = True
htmlhelp_basename = 'nengo_mpidoc'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo_mpi.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo_mpi', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo_mpi', html_title, authors, 'nengo_mpi',
     'MPI backend for the Nengo neural simulation library', 'Miscellaneous'),
]
