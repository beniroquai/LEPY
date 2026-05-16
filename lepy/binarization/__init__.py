"""
Binarisation back-ends for LEPY.

This file is identical to upstream LEPY except that ``unet`` and ``flatbug``
are imported lazily inside :func:`binarize`. They depend on the optional
``backgroundremover`` / ``flat-bug`` packages, and the napari-lepy plugin
must remain usable when only ``otsu`` / ``gauss_local`` / ``grabcut`` are
available.

Public API (function names, behaviour) is unchanged.
"""
from lepy.binarization.otsu import otsu
from lepy.binarization.gauss_local import gauss_local
from lepy.binarization.grabcut import stable_grabcut
from lepy.binarization.grabcut import grabcut


def binarize(im, chan, method: str):

    if method == "otsu":
        return otsu(chan)

    elif method == "gauss_local":
        return gauss_local(chan)

    elif method.startswith("grabcut"):
        return stable_grabcut(im, chan, n_rounds=5,
                              plus="+" in method,
                              use_otsu="otsu" in method,
                              rescale_factor=1/4)

    elif method == "unet":
        # Lazy import: requires `backgroundremover` (optional dep).
        from lepy.binarization.unet import unet
        return unet(im)

    elif method == "flatbug":
        # Lazy import: requires `flat-bug` (optional dep).
        from lepy.binarization.flatbug import flatbug
        return flatbug(im)

    else:
        raise ValueError(f"Unknown method: {method}!")


__all__ = [
    "binarize",
    "otsu",
    "gauss_local",
    "grabcut",
    "stable_grabcut",
]
