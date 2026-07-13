"""Sphinx configuration for the API documentation.

Build:  pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build
Heavy runtime deps (torch, transformers, ...) are mocked so the docs build on a
plain CPU runner without the model.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Qwen3 Circuit Reproduction"
author = "Iuliia Vitiugova"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# mock the heavy / GPU dependencies so autodoc can import the modules
autodoc_mock_imports = [
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "networkx",
    "sklearn",
    "scipy",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
