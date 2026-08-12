"""
Colour / symbol scheme for the eight LEPY points of interest.

Design rationale (default scheme ``role_symbol``)
-------------------------------------------------
The two hard cases when reviewing are ``inner_top_l`` vs ``inner_top_r`` and
``inner_bot_l`` vs ``inner_bot_r`` -- they sit only a few dozen pixels apart
next to the body and frequently overlap on screen.  The default scheme
therefore encodes the information twice:

* **marker symbol = anatomical role** (body / outer / inner-top / inner-bot),
  which stays readable even when two markers are drawn on top of each other or
  the screen is greyscale,
* **hue = the individual point**, with maximally contrasting hues inside each
  overlapping pair (teal vs magenta, olive vs purple).

All palettes are taken from Paul Tol's / Okabe-Ito's qualitative sets and are
safe for deuteranopia, protanopia and tritanopia.
"""
from __future__ import annotations

import typing as T
from dataclasses import dataclass


#: Canonical POI order.  Identical to ``PointsOfInterest.__iter__`` (minus
#: ``center``) and to the order used by the napari Points layer.
POI_NAMES: T.List[str] = [
    "body_top", "body_bot",
    "outer_l", "outer_r",
    "inner_top_l", "inner_top_r",
    "inner_bot_l", "inner_bot_r",
]

#: Human readable labels shown in the widget's legend.
POI_LABELS: T.Dict[str, str] = {
    "body_top":    "Body top",
    "body_bot":    "Body bottom",
    "outer_l":     "Outer left",
    "outer_r":     "Outer right",
    "inner_top_l": "Inner top left",
    "inner_top_r": "Inner top right",
    "inner_bot_l": "Inner bottom left",
    "inner_bot_r": "Inner bottom right",
}

#: Short labels drawn next to the markers in the canvas.
POI_SHORT: T.Dict[str, str] = {
    "body_top":    "b.top",
    "body_bot":    "b.bot",
    "outer_l":     "out.L",
    "outer_r":     "out.R",
    "inner_top_l": "in.top.L",
    "inner_top_r": "in.top.R",
    "inner_bot_l": "in.bot.L",
    "inner_bot_r": "in.bot.R",
}

#: Anatomical role each POI belongs to -- used for grouping in the legend.
POI_ROLE: T.Dict[str, str] = {
    "body_top":    "body",
    "body_bot":    "body",
    "outer_l":     "outer",
    "outer_r":     "outer",
    "inner_top_l": "inner_top",
    "inner_top_r": "inner_top",
    "inner_bot_l": "inner_bot",
    "inner_bot_r": "inner_bot",
}


@dataclass(frozen=True)
class PoiStyle:
    """How one POI is drawn in the napari Points layer."""

    color: str
    symbol: str


#: ``symbol = role, hue = identity`` -- Paul Tol "vibrant"/"bright".  Default.
_ROLE_SYMBOL: T.Dict[str, PoiStyle] = {
    "body_top":    PoiStyle("#EE7733", "triangle_up"),    # orange
    "body_bot":    PoiStyle("#CC3311", "triangle_down"),  # red
    "outer_l":     PoiStyle("#0077BB", "disc"),           # blue
    "outer_r":     PoiStyle("#33BBEE", "disc"),           # cyan
    "inner_top_l": PoiStyle("#009988", "square"),         # teal
    "inner_top_r": PoiStyle("#EE3377", "square"),         # magenta
    "inner_bot_l": PoiStyle("#BBBB00", "diamond"),        # olive
    "inner_bot_r": PoiStyle("#AA4499", "diamond"),        # purple
}

#: ``hue = role, symbol = side`` -- Okabe-Ito.  Fewer colours to memorise.
_ROLE_HUE: T.Dict[str, PoiStyle] = {
    "body_top":    PoiStyle("#E69F00", "triangle_up"),
    "body_bot":    PoiStyle("#E69F00", "triangle_down"),
    "outer_l":     PoiStyle("#0072B2", "disc"),
    "outer_r":     PoiStyle("#0072B2", "square"),
    "inner_top_l": PoiStyle("#009E73", "disc"),
    "inner_top_r": PoiStyle("#009E73", "square"),
    "inner_bot_l": PoiStyle("#D55E00", "disc"),
    "inner_bot_r": PoiStyle("#D55E00", "square"),
}

#: Eight distinct hues, uniform discs -- Paul Tol "muted".
_FLAT: T.Dict[str, PoiStyle] = {
    "body_top":    PoiStyle("#332288", "disc"),
    "body_bot":    PoiStyle("#88CCEE", "disc"),
    "outer_l":     PoiStyle("#CC6677", "disc"),
    "outer_r":     PoiStyle("#882255", "disc"),
    "inner_top_l": PoiStyle("#44AA99", "disc"),
    "inner_top_r": PoiStyle("#117733", "disc"),
    "inner_bot_l": PoiStyle("#DDCC77", "disc"),
    "inner_bot_r": PoiStyle("#AA4499", "disc"),
}

SCHEMES: T.Dict[str, T.Dict[str, PoiStyle]] = {
    "role_symbol": _ROLE_SYMBOL,
    "role_hue": _ROLE_HUE,
    "flat": _FLAT,
}

SCHEME_LABELS: T.Dict[str, str] = {
    "role_symbol": "Symbol = role, hue = point (Tol, CVD-safe)",
    "role_hue": "Hue = role, symbol = side (Okabe-Ito)",
    "flat": "8 hues, all discs (Tol muted)",
}

DEFAULT_SCHEME = "role_symbol"


def styles(scheme: str = DEFAULT_SCHEME) -> T.Dict[str, PoiStyle]:
    """Return the ``{poi_name: PoiStyle}`` mapping for *scheme*."""
    return SCHEMES.get(scheme, SCHEMES[DEFAULT_SCHEME])


def colors(scheme: str = DEFAULT_SCHEME) -> T.List[str]:
    """Hex colours in :data:`POI_NAMES` order."""
    st = styles(scheme)
    return [st[n].color for n in POI_NAMES]


def symbols(scheme: str = DEFAULT_SCHEME) -> T.List[str]:
    """napari marker symbols in :data:`POI_NAMES` order."""
    st = styles(scheme)
    return [st[n].symbol for n in POI_NAMES]


#: Kept for backwards compatibility with :mod:`napari_lepy._layer_io`.
POI_COLORS: T.Dict[str, str] = {n: s.color for n, s in _ROLE_SYMBOL.items()}
