# -*- coding: utf-8 -*-

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = u'Essentia'
copyright = u'2006-2024, Universitat Pompeu Fabra'
author = 'MTG'

release = '2.1-beta6-dev'
version = '2.1-beta6-dev'
root_doc = 'documentation'

# -- General configuration

extensions = [
    'sphinx.ext.viewcode',
    'sphinxcontrib.doxylink',
    'sphinx.ext.autosectionlabel',
    'sphinx_toolbox.collapse',
    'sphinx_copybutton',
    ]

# -- Options for HTML output
templates_path = ['_templates']
exclude_patterns = ['_build']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_logo = "_static/essentia_logo.svg"
html_favicon = '_static/favicon.ico'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Sidebar templates
# html_sidebars = {
#     '**': []
# }

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
                        #  'index': 'index.html',
                        #  'applications': 'applications.html',
                        #  'documentation': 'documentation.html',
                         }

# html_extra_path = ['./_templates/index.html']

doxylink = {
    'essentia': ('EssentiaDoxygen.tag', 'doxygen')
}
