"""
Recompute LEPY statistics after a manual correction.

Everything here funnels through the *original* LEPY code
(:class:`lepy.poi.PointsOfInterest`, :mod:`lepy.segmentation`,
``lepy.Image.color_stats``) so that corrected rows are numerically consistent
with rows produced by an unattended LepyLoop run -- including LEPY's quirks,
such as ``PointsOfInterest.distances`` quantising to whole millimetres before
measuring.

Two kinds of correction are distinguished:

``POI only``
    Only ``poi_*`` entries change.  ``lepy`` derives the POI areas from the
    *contour*, which is unchanged, so nothing else has to be touched.

``mask / contour changed``
    ``contour_*`` and -- because the mask selects which pixels enter the
    histograms -- every colour statistic changes too.  Recomputing the colour
    block needs the full-resolution image and is therefore opt-in.
"""
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lepy.outputs import OUTPUTS as OUTS
from lepy.poi import PointsOfInterest

from napari_lepy.review.palette import POI_NAMES

logger = logging.getLogger(__name__)


#: Keys that only depend on the POIs.
POI_KEYS: T.List[str] = [
    OUTS.poi.orig_width, OUTS.poi.orig_height,
    OUTS.poi.center.x, OUTS.poi.center.y,
    OUTS.poi.dist.inner_outer_l, OUTS.poi.dist.inner_outer_r,
    OUTS.poi.dist.inner, OUTS.poi.dist.body,
    OUTS.poi.area.body, OUTS.poi.area.wing_l, OUTS.poi.area.wing_r,
] + [getattr(getattr(OUTS.poi, name), axis)
     for name in POI_NAMES for axis in ("x", "y")]

#: Keys that depend on the contour / mask.
CONTOUR_KEYS: T.List[str] = [
    OUTS.contour.length, OUTS.contour.area,
    OUTS.contour.xmin, OUTS.contour.xmax, OUTS.contour.ymin, OUTS.contour.ymax,
    OUTS.contour.area_calibrated,
    OUTS.contour.width_calibrated, OUTS.contour.height_calibrated,
]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def rasterize_contour(contour_px: np.ndarray,
                      shape: T.Tuple[int, int]) -> np.ndarray:
    """Fill a LEPY contour ``(N, 2) (x, y)`` into a ``uint8`` {0, 1} mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if contour_px is None or len(contour_px) < 3:
        return mask
    pts = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.drawContours(mask, [pts], 0, 1, -1)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    """Largest external contour of a binary mask, LEPY's ``(N, 2) (x, y)``."""
    if mask is None:
        return np.zeros((0, 2), dtype=np.int32)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.int32)
    largest = max(contours, key=cv2.contourArea)
    return largest[:, 0].astype(np.int32)


def simplify_contour(contour_px: np.ndarray, epsilon_px: float) -> np.ndarray:
    """Douglas-Peucker simplification -- makes vertex editing usable.

    A LEPY contour routinely has >2000 vertices; dragging individual vertices
    of a napari polygon that large is painfully slow.
    """
    if contour_px is None or len(contour_px) < 4 or epsilon_px <= 0:
        return np.asarray(contour_px, dtype=np.int32)
    pts = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
    reduced = cv2.approxPolyDP(pts, float(epsilon_px), True)
    if len(reduced) < 3:
        return np.asarray(contour_px, dtype=np.int32)
    return reduced[:, 0].astype(np.int32)


def contour_to_polygon(contour_px: np.ndarray) -> np.ndarray:
    """LEPY ``(x, y)`` contour -> napari polygon vertices ``(row, col)``."""
    if contour_px is None or len(contour_px) == 0:
        return np.zeros((0, 2), dtype=float)
    contour_px = np.asarray(contour_px).reshape(-1, 2)
    return np.column_stack([contour_px[:, 1], contour_px[:, 0]]).astype(float)


def polygon_to_contour(polygon: np.ndarray) -> np.ndarray:
    """napari polygon ``(row, col)`` -> LEPY ``(x, y)`` contour."""
    if polygon is None or len(polygon) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    polygon = np.asarray(polygon).reshape(-1, 2)
    return np.round(
        np.column_stack([polygon[:, 1], polygon[:, 0]])).astype(np.int32)


def keypoints_from_rowcol(poi_rowcol: np.ndarray) -> T.Dict[str, int]:
    """``(8, 2)`` ``(row, col)`` array -> LEPY ``key_points`` dict.

    Uses the official column names from :mod:`lepy.outputs.definitions`, i.e.
    exactly what :meth:`PointsOfInterest.from_keypoints` expects.
    """
    poi_rowcol = np.asarray(poi_rowcol, dtype=float)
    if poi_rowcol.shape != (len(POI_NAMES), 2):
        raise ValueError(
            f"expected {len(POI_NAMES)} POIs as (row, col), got {poi_rowcol.shape}")
    keypoints: T.Dict[str, int] = {}
    for index, name in enumerate(POI_NAMES):
        coord = getattr(OUTS.poi, name)
        keypoints[coord.y] = int(round(poi_rowcol[index, 0]))
        keypoints[coord.x] = int(round(poi_rowcol[index, 1]))
    return keypoints


def rowcol_from_stats(stats: T.Mapping[str, T.Any]) -> T.Optional[np.ndarray]:
    """Inverse of :func:`keypoints_from_rowcol`, reading a stats dict."""
    rows = []
    for name in POI_NAMES:
        coord = getattr(OUTS.poi, name)
        x, y = stats.get(coord.x), stats.get(coord.y)
        if x is None or y is None:
            return None
        try:
            rows.append([float(y), float(x)])
        except (TypeError, ValueError):
            return None
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def poi_stats(contour_px: np.ndarray, poi_rowcol: np.ndarray,
              shape: T.Tuple[int, int],
              scale: T.Optional[float] = None) -> T.Dict[str, T.Any]:
    """Recompute the ``poi_*`` block for hand-placed POIs.

    Mirrors :meth:`lepy.Image.pois` exactly: the binary image the POI code sees
    is the *filled contour*, not the segmentation mask.
    """
    filled = rasterize_contour(contour_px, shape)
    if not filled.any():
        raise ValueError("contour is empty -- cannot recompute POI statistics")

    keypoints = keypoints_from_rowcol(poi_rowcol)
    pois = PointsOfInterest.from_keypoints(filled, key_points=keypoints)

    out: T.Dict[str, T.Any] = {}
    out.update(pois.stats)
    out.update(pois.distances(scale=scale))
    out.update(pois.areas(bin_im=filled, scale=scale))
    return out


#: Suffix of the extra, un-quantised distance columns written to ``stats_json``.
EXACT_SUFFIX = "_exact"


def exact_distances(poi_rowcol: np.ndarray,
                    scale: T.Optional[float] = None) -> T.Dict[str, float]:
    """Euclidean POI distances *without* LEPY's integer-millimetre rounding.

    ``lepy.poi.Point.__post_init__`` casts row/col to ``int``, and
    ``PointsOfInterest.distances`` divides by the calibration *before*
    measuring -- so the official ``poi_dist_*`` values are computed on
    coordinates truncated to whole millimetres.  At the ~166 px/mm of this
    dataset that is a ~1 mm grid, which is coarse relative to a 6 mm forewing
    and means small POI corrections do not move the number at all.

    These extra ``*_exact`` keys measure in pixels first and convert
    afterwards.  They are written alongside the official columns so the two can
    be compared; nothing downstream is forced to use them.
    """
    poi_rowcol = np.asarray(poi_rowcol, dtype=float)
    index = {name: i for i, name in enumerate(POI_NAMES)}
    divisor = float(scale) if scale and scale > 0 else 1.0
    pairs = (
        (OUTS.poi.dist.inner, "inner_top_l", "inner_top_r"),
        (OUTS.poi.dist.body, "body_top", "body_bot"),
        (OUTS.poi.dist.inner_outer_l, "inner_top_l", "outer_l"),
        (OUTS.poi.dist.inner_outer_r, "inner_top_r", "outer_r"),
    )
    out: T.Dict[str, float] = {}
    for key, first, second in pairs:
        delta = poi_rowcol[index[first]] - poi_rowcol[index[second]]
        out[f"{key}{EXACT_SUFFIX}"] = float(np.hypot(*delta) / divisor)
    return out


def contour_stats(contour_px: np.ndarray,
                  scale: T.Optional[float] = None) -> T.Dict[str, T.Any]:
    """Recompute the ``contour_*`` block, matching :func:`lepy.segmentation.segment`."""
    contour_px = np.asarray(contour_px).reshape(-1, 2).astype(np.int32)
    if len(contour_px) < 3:
        raise ValueError("contour needs at least three vertices")

    out: T.Dict[str, T.Any] = {
        OUTS.contour.length: len(contour_px),
        OUTS.contour.area: float(cv2.contourArea(contour_px.reshape(-1, 1, 2))),
        OUTS.contour.xmin: int(np.amin(contour_px[:, 0])),
        OUTS.contour.xmax: int(np.amax(contour_px[:, 0])),
        OUTS.contour.ymin: int(np.amin(contour_px[:, 1])),
        OUTS.contour.ymax: int(np.amax(contour_px[:, 1])),
    }
    if scale and scale > 0:
        out[OUTS.contour.area_calibrated] = out[OUTS.contour.area] / scale ** 2
        out[OUTS.contour.width_calibrated] = \
            (out[OUTS.contour.xmax] - out[OUTS.contour.xmin]) / scale
        out[OUTS.contour.height_calibrated] = \
            (out[OUTS.contour.ymax] - out[OUTS.contour.ymin]) / scale
    return out


def color_stats(rgb: np.ndarray, mask: np.ndarray,
                uv: T.Optional[np.ndarray] = None,
                binsize: int = 2) -> T.Dict[str, T.Any]:
    """Recompute the colour / hue / saturation / intensity blocks.

    Delegates to :meth:`lepy.Image.color_stats` and
    :func:`lepy.segmentation.segment`'s HSV block via a throw-away
    :class:`lepy.Image` shell, so the numbers stay identical to a real run.
    """
    from lepy import Image as LepyImage

    image = LepyImage(root=None, key="napari", rgb_fname=None)
    image.rgb_im = rgb
    image.gray_im = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    image.uv_im = uv
    image.mask = (np.asarray(mask) > 0).astype(np.uint8)
    image.stats = {}
    image.color_stats(binsize=binsize)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    out = dict(image.stats)
    out.update(OUTS.hue.calc_stats(hue, mask=image.mask))
    out.update(OUTS.saturation.calc_stats(saturation, mask=image.mask))
    out.update(OUTS.intensity.calc_stats(value, mask=image.mask))
    return out


def recompute(base_stats: T.Mapping[str, T.Any], *,
              contour_px: np.ndarray,
              poi_rowcol: np.ndarray,
              shape: T.Tuple[int, int],
              scale: T.Optional[float] = None,
              mask_changed: bool = False,
              rgb: T.Optional[np.ndarray] = None,
              uv: T.Optional[np.ndarray] = None,
              color_mask: T.Optional[np.ndarray] = None,
              recompute_color: bool = False) -> T.Dict[str, T.Any]:
    """Return a full stats dict with the corrected values merged in.

    Parameters
    ----------
    base_stats
        The original LepyLoop stats for this specimen; untouched keys survive.
    mask_changed
        Whether the contour was edited.  Only then are ``contour_*`` (and,
        with *recompute_color*, the colour blocks) redone.
    color_mask
        Which pixels enter the colour histograms.  Pass the brush mask when the
        user painted one -- it can have holes, while a filled polygon cannot.
        Defaults to the filled contour.
    recompute_color
        Recompute the colour histograms from *rgb*.  Costs a second or two on
        a 6000x4000 image, so the caller decides.
    """
    stats: T.Dict[str, T.Any] = dict(base_stats)
    height, width = shape
    stats[OUTS.image.width] = int(width)
    stats[OUTS.image.height] = int(height)

    if mask_changed:
        stats.update(contour_stats(contour_px, scale=scale))
        if recompute_color and rgb is not None:
            mask = color_mask if color_mask is not None \
                else rasterize_contour(contour_px, shape)
            try:
                stats.update(color_stats(rgb, mask, uv=uv))
            except Exception:
                logger.exception("Colour statistics could not be recomputed")

    stats.update(poi_stats(contour_px, poi_rowcol, shape, scale=scale))
    stats.update(exact_distances(poi_rowcol, scale=scale))
    return stats


def poi_shift(original: T.Optional[np.ndarray],
              current: np.ndarray,
              tolerance_px: float = 0.5) -> T.Tuple[int, float]:
    """How far the POIs moved: ``(n_points_moved, largest_shift_px)``."""
    if original is None or current is None:
        return 0, 0.0
    original = np.asarray(original, dtype=float)
    current = np.asarray(current, dtype=float)
    if original.shape != current.shape:
        return len(current), float("inf")
    distances = np.linalg.norm(current - original, axis=1)
    moved = int(np.sum(distances > tolerance_px))
    return moved, float(distances.max()) if distances.size else 0.0
