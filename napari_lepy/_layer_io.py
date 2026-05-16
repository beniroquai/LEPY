"""
Bridge helpers between napari layer state and the original LEPY data
structures.

The goal is: use LEPY classes and column names verbatim, only translate to /
from numpy arrays where napari needs them.
"""
from __future__ import annotations

import typing as T
import numpy as np

from lepy.outputs import OUTPUTS as OUTS
from lepy.poi import PointsOfInterest, Point


# Order in which POIs are stored in the napari Points layer.
# Identical to PointsOfInterest.__iter__ (excluding `center`).
POI_NAMES: T.List[str] = [
    "body_top", "body_bot",
    "outer_l", "outer_r",
    "inner_top_l", "inner_top_r",
    "inner_bot_l", "inner_bot_r",
]

# Per-POI colour for the napari Points layer (matches LEPY's pipeline diagram).
POI_COLORS = {
    "body_top":    "cyan",
    "body_bot":    "cyan",
    "outer_l":     "magenta",
    "outer_r":     "magenta",
    "inner_top_l": "yellow",
    "inner_top_r": "yellow",
    "inner_bot_l": "orange",
    "inner_bot_r": "orange",
}


def pois_to_array(pois: PointsOfInterest) -> np.ndarray:
    """Convert a ``PointsOfInterest`` into an (N, 2) ``[row, col]`` array
    in the order of :data:`POI_NAMES` -- the order napari expects for a
    Points layer.
    """
    rows = []
    for name in POI_NAMES:
        p: Point = getattr(pois, name)
        rows.append([p.row, p.col])
    return np.asarray(rows, dtype=float)


def array_to_keypoints(arr: np.ndarray) -> dict:
    """Convert an (N, 2) ``[row, col]`` array (as edited in the napari Points
    layer) into the ``key_points`` dict that
    :meth:`lepy.poi.PointsOfInterest.from_keypoints` expects.

    The dict uses LEPY's official column names from
    :mod:`lepy.outputs.definitions` (``poi_body_top_x``, ``poi_body_top_y``,
    etc.) so nothing downstream needs to change.
    """
    if arr.shape[0] != len(POI_NAMES):
        raise ValueError(
            f"POI array must have exactly {len(POI_NAMES)} rows, got {arr.shape[0]}"
        )
    kp: dict = {}
    for i, name in enumerate(POI_NAMES):
        coord = getattr(OUTS.poi, name)
        kp[coord.y] = int(round(arr[i, 0]))  # row -> y
        kp[coord.x] = int(round(arr[i, 1]))  # col -> x
    return kp


def labels_to_mask(layer_data: np.ndarray) -> np.ndarray:
    """Convert napari Labels layer data (any integer dtype) to LEPY's mask
    convention: ``uint8`` with values in ``{0, 1}``."""
    return (np.asarray(layer_data) > 0).astype(np.uint8)


def contour_to_polygon(contour: np.ndarray) -> np.ndarray:
    """Convert a LEPY contour ``(N, 2)`` in ``(x, y)`` to a napari polygon
    in ``(row, col)``."""
    if contour is None or len(contour) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([contour[:, 1], contour[:, 0]]).astype(float)


def recompute_contour(mask: np.ndarray) -> np.ndarray:
    """Recompute the largest external contour from a (possibly edited) binary
    mask -- returned in LEPY's ``(N, 2) (x, y)`` convention.
    """
    import cv2
    if mask is None:
        return np.zeros((0, 2), dtype=np.int32)
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros((0, 2), dtype=np.int32)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0][:, 0]  # (N, 2) (x, y)
