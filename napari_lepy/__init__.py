"""
napari-lepy
===========

Interactive napari plugin wrapper around the original LEPY pipeline.

This package does **not** reimplement LEPY. It imports `lepy.*` directly and
adds a thin interactive layer that lets the user:

    1. Run each LEPY step inside napari (loaded as image / labels layers).
    2. Edit the binary mask by hand using napari's brush / eraser.
    3. Re-run POI detection (``lepy.poi.PointsOfInterest.detect``) on the
       edited mask without leaving napari.
    4. Drag any POI to a new location -- subsequent traits are recomputed via
       the original ``PointsOfInterest.from_keypoints`` constructor.
    5. Export every layer + the original LEPY CSV/JSON outputs.

It also ships a second widget, *LEPY Review & Correction*
(:mod:`napari_lepy.review`), for the batch workflow: step through a finished
LepyLoop run, confirm or reject its results, fix the wrong ones in place, and
hand the corrected subset back to LepyLoop.

The original LEPY API surface (``lepy.Image``, ``lepy.Worker``,
``lepy.PointsOfInterest``, ``lepy.OutputWriter``, ``lepy.Plotter``) is
re-exported here unchanged.
"""

from lepy import (
    Image,
    PointsOfInterest,
    Worker,
    OutputWriter,
    Plotter,
    segment,
)
from lepy import __version__ as _lepy_version

__version__ = "0.1.0"
__lepy_version__ = _lepy_version

__all__ = [
    "Image",
    "PointsOfInterest",
    "Worker",
    "OutputWriter",
    "Plotter",
    "segment",
]
