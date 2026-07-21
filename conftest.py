"""Pytest bootstrap: make the src/ package modules importable.

The application modules (stitcher, modality, envi_io, ortho, gui) live in src/
and import each other by flat name (e.g. ``import stitcher``). Adding src/ to
sys.path here lets the test suite import them the same way the packaged app and
the ``python src/gui.py`` entry point do.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
