"""Sphinx configuration for raft-lm API documentation."""

from __future__ import annotations

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

project = "raft-lm"
author = "Risk-Aware AI"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "vault", "project-plan-docs"]

html_theme = "furo"
html_logo = "../assets/logo.png"
html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#7c9cff",
        "color-brand-content": "#e8eaed",
    },
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
