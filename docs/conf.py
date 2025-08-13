# Configuration file for the Sphinx documentation builder.
import os, sys
sys.path.insert(0, os.path.abspath('../')) 


# -- Project information -----------------------------------------------------

project = 'kbkit'
copyright = '2025, Allison A. Peroutka'
author = 'Allison A. Peroutka'
release = '0.0.1'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_copybutton',
    'numpydoc'
]
napolean_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']

autodoc_member_order = 'bysource'  # options: 'bysource', 'groupwise'

html_theme = 'furo' # nature

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#6851ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a08cff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/aperoutka/kbkit/",
    "source_branch": "main",
    "source_directory": "docs",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/aperoutka/kbkit/",
            "fa": "fa-brands fa-github",
        },
    ],
    "announcement": "Check out our latest release!",
}

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

nbsphinx_allow_errors = True