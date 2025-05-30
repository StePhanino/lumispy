# Configuration file for the Sphinx documentation builder.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from datetime import datetime
from importlib.metadata import version as get_version

sys.path.append("../")


# -- Project information
project = "LumiSpy"
copyright = f"2019-{datetime.today().year}, The LumiSpy developers"
author = "The LumiSpy developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = get_version("lumispy")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.autosummary",
    "sphinxcontrib.towncrier",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "exspy": ("https://hyperspy.org/exspy/", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current/", None),
    "kikuchipy": ("https://kikuchipy.org/en/latest/", None),
    "rsciio": ("https://hyperspy.org/rosettasciio/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_disabled_domains = ["std"]

linkcheck_ignore = [
    "https://doi.org/10.1063/5.0080486",  # 403 Client Error: Forbidden for url
    "https://doi.org/10.1021/jz401508t",  # 403 Client Error: Forbidden for url
    "https://github.com/LumiSpy/lumispy/security/code-scanning",  # 404 Client Error: Not Found for url (even though page exists)
]

# imgmath: Sphinx allows use of LaTeX in the html documentation, but not directly. It is first rendered to an image.
# You can add here whatever preamble you are used to adding to your LaTeX document.
imgmath_latex_preamble = r"""
    \usepackage{xcolor}
    \definecolor{mathcolor}{rgb}{0.8,0.3,0.1}
    \everymath{\color{mathcolor}}
    %\everydisplay{\color{mathcolor}}
"""


templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo_rec_april21.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/lumispy.ico"

# -- Options for EPUB output
epub_show_urls = "footnote"


def run_apidoc(_):
    # https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
    # https://www.sphinx-doc.org/es/1.2/ext/autodoc.html
    import os

    os.environ["SPHINX_APIDOC_OPTIONS"] = (
        "members,private-members,no-undoc-members,show-inheritance,ignore-module-all"
    )

    from sphinx.ext.apidoc import main

    cur_dir = os.path.normpath(os.path.dirname(__file__))
    output_path = os.path.join(cur_dir, "api")
    modules = os.path.normpath(os.path.join(cur_dir, "../lumispy"))
    exclude_pattern = [
        "../lumispy/tests",
        "../lumispy/components",
        "../lumispy/release_info.py",
    ]
    main(["-e", "-f", "-P", "-o", output_path, modules, *exclude_pattern])


def setup(app):
    app.connect("builder-inited", run_apidoc)
    app.add_css_file("css/dark.css")
    app.add_css_file("css/light.css")


# CopyButton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"


# -- Options for towncrier_draft extension -----------------------------------

# Options: draft/sphinx-version/sphinx-release
towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = False
towncrier_draft_working_directory = ".."
