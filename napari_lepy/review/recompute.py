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
    ``contour_*`` (area net of any holes) and -- because the mask selects which pixels enter the
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
Holes = T.Sequence[np.ndarray]


def _valid_holes(holes: T.Optional[Holes]) -> T.List[np.ndarray]:
    return [np.asarray(h, dtype=np.int32).reshape(-1, 2)
            for h in (holes or []) if h is not None and len(h) >= 3]


def rasterize_contour(contour_px: np.ndarray,
                      shape: T.Tuple[int, int],
                      holes: T.Optional[Holes] = None) -> np.ndarray:
    """Fill a LEPY contour ``(N, 2) (x, y)`` into a ``uint8`` {0, 1} mask.

    *holes* are further ``(x, y)`` polygons that are cut out again, e.g.
    background enclosed by an antenna or leg that touches the body twice.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if contour_px is None or len(contour_px) < 3:
        return mask
    pts = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
    mask = cv2.drawContours(mask, [pts], 0, 1, -1)
    for hole in _valid_holes(holes):
        mask = cv2.drawContours(mask, [hole.reshape(-1, 1, 2)], 0, 0, -1)
    return mask


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


def holes_from_mask(mask: np.ndarray,
                    min_area_px: float = 0.0) -> T.List[np.ndarray]:
    """Holes of the largest object in a binary mask, each ``(N, 2) (x, y)``.

    The polygons trace the *background* pixels of each hole, so
    ``rasterize_contour(outer, shape, holes)`` reproduces the mask exactly
    instead of eating a one-pixel rim of the animal.  Holes smaller than
    *min_area_px* (brush speckles) are dropped.
    """
    outer = contour_from_mask(mask)
    if len(outer) < 3:
        return []
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    enclosed = rasterize_contour(outer, binary.shape) & (1 - binary)
    contours, _ = cv2.findContours(enclosed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0].astype(np.int32) for c in contours
            if len(c) >= 3 and cv2.contourArea(c) >= min_area_px]


def _odd(n: float) -> int:
    return max(3, int(round(n)) | 1)


def detect_background_holes(rgb: np.ndarray, contour_px: np.ndarray,
                            holes: T.Optional[Holes] = None,
                            tolerance: float = 3.0,
                            min_area_frac: float = 1e-4,
                            appendage_frac: float = 0.015,
                            work_px: int = 1500) -> T.List[np.ndarray]:
    """Suggest holes: background that the outline wrongly encloses.

    Typical case: an antenna or leg touches the body twice and the outline
    closes around a patch of background.

    1. The background colour is taken from a ring just outside the outline;
       pixels inside whose Lab distance to it is below ``tolerance`` x the
       ring's own spread are candidates.
    2. Pale wing patches can match the background colour, so a candidate is
       only kept if it is closed off by *thin* structures: after a
       morphological opening of the animal with a kernel of
       ``appendage_frac`` x the specimen size (removing antennae and legs) it
       must be connected to the outside.
    3. Survivors are traced again at full resolution.

    Steps 1-2 run on a copy downscaled to ``work_px``.  Components smaller
    than ``min_area_frac`` of the image are dropped.  These are suggestions
    for the user to check, not ground truth.
    """
    shape = rgb.shape[:2]
    contour_px = np.asarray(contour_px, dtype=np.int32).reshape(-1, 2)
    if len(contour_px) < 3:
        return []
    holes = _valid_holes(holes)
    area_min = float(min_area_frac) * shape[0] * shape[1]

    # ---- low resolution: colour candidates + "closed off by thin parts" test
    f = min(1.0, work_px / max(shape))
    small_shape = (max(1, int(round(shape[0] * f))), max(1, int(round(shape[1] * f))))
    small = cv2.resize(rgb, small_shape[::-1], interpolation=cv2.INTER_AREA)
    scaled = lambda c: np.round(c * f).astype(np.int32)  # noqa: E731
    outer_s = rasterize_contour(scaled(contour_px), small_shape)
    inside_s = rasterize_contour(scaled(contour_px), small_shape,
                                 [scaled(h) for h in holes])
    if not inside_s.any():
        return []
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(0.01 * max(small_shape)),) * 2)
    ring = cv2.dilate(outer_s, ring_k, iterations=3) & (
        1 - cv2.dilate(outer_s, ring_k))
    if ring.sum() < 50:
        return []
    lab_s = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = lab_s[ring > 0]
    median = np.median(ref, axis=0)
    threshold = max(tolerance * float(np.median(
        np.linalg.norm(ref - median, axis=1))), 4.0)

    def candidates(lab: np.ndarray, inside: np.ndarray) -> np.ndarray:
        near = np.linalg.norm(lab - median, axis=2) < threshold
        cand = (near & (inside > 0)).astype(np.uint8)
        return cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3)

    # the eroded interior drops slivers between the polygon and the animal edge
    cand_s = candidates(lab_s, cv2.erode(inside_s, k3, iterations=2))
    if not cand_s.any():
        return []

    x, y = contour_px[:, 0], contour_px[:, 1]
    size = max(x.max() - x.min(), y.max() - y.min()) * f
    thick = cv2.morphologyEx(
        inside_s & (1 - cand_s), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_odd(appendage_frac * size),) * 2))
    _, free = cv2.connectedComponents((thick == 0).astype(np.uint8))
    open_ids = np.unique(free[outer_s == 0])

    n, comps = cv2.connectedComponents(cand_s)
    keep_s = np.zeros(small_shape, np.uint8)
    for i in range(1, n):
        comp = comps == i
        if comp.sum() < area_min * f * f:
            continue
        if np.isin(free[comp], open_ids).mean() > 0.5:
            keep_s[comp] = 1
    if not keep_s.any():
        return []

    # ---- full resolution: re-trace the accepted regions precisely
    roi = cv2.resize(cv2.dilate(keep_s, k3, iterations=3), shape[::-1],
                     interpolation=cv2.INTER_NEAREST)
    x0, y0, w, h = cv2.boundingRect(roi)
    sub = (slice(y0, y0 + h), slice(x0, x0 + w))
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[sub]),
                       cv2.COLOR_RGB2LAB).astype(np.float32)
    inside = rasterize_contour(contour_px, shape, holes)[sub]
    cand = candidates(lab, inside) & roi[sub]
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(int(x0), int(y0)))
    return [c[:, 0].astype(np.int32) for c in contours
            if cv2.contourArea(c) >= area_min]


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
              scale: T.Optional[float] = None,
              holes: T.Optional[Holes] = None) -> T.Dict[str, T.Any]:
    """Recompute the ``poi_*`` block for hand-placed POIs.

    Mirrors :meth:`lepy.Image.pois` exactly: the binary image the POI code sees
    is the *filled contour*, not the segmentation mask -- minus any *holes*.
    """
    filled = rasterize_contour(contour_px, shape, holes)
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


#: Extra keys (stats_json only) describing the holes cut into the outline.
HOLE_COUNT_KEY = "contour_holes"
HOLE_AREA_KEY = "contour_holes_area"


def contour_stats(contour_px: np.ndarray,
                  scale: T.Optional[float] = None,
                  holes: T.Optional[Holes] = None,
                  shape: T.Optional[T.Tuple[int, int]] = None
                  ) -> T.Dict[str, T.Any]:
    """Recompute the ``contour_*`` block, matching :func:`lepy.segmentation.segment`.

    With *holes* (and *shape*), ``contour_area`` is the outline's area minus
    the pixels the holes remove -- measured by rasterising, so overlapping or
    partly-outside holes are not double counted.  Length and bounding box
    stay those of the outer outline.
    """
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
    holes = _valid_holes(holes)
    hole_area = 0.0
    if holes and shape is not None:
        hole_area = float(rasterize_contour(contour_px, shape).sum()
                          - rasterize_contour(contour_px, shape, holes).sum())
        out[OUTS.contour.area] = max(0.0, out[OUTS.contour.area] - hole_area)
    out[HOLE_COUNT_KEY] = len(holes)
    out[HOLE_AREA_KEY] = hole_area
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
              recompute_color: bool = False,
              holes: T.Optional[Holes] = None) -> T.Dict[str, T.Any]:
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
    holes
        Polygons cut out of the outline (background enclosed by antennae or
        legs).  They reduce the contour / POI areas and the colour mask.
    """
    stats: T.Dict[str, T.Any] = dict(base_stats)
    height, width = shape
    stats[OUTS.image.width] = int(width)
    stats[OUTS.image.height] = int(height)

    if mask_changed:
        stats.update(contour_stats(contour_px, scale=scale, holes=holes,
                                   shape=shape))
        if recompute_color and rgb is not None:
            mask = color_mask if color_mask is not None \
                else rasterize_contour(contour_px, shape, holes)
            try:
                stats.update(color_stats(rgb, mask, uv=uv))
            except Exception:
                logger.exception("Colour statistics could not be recomputed")

    stats.update(poi_stats(contour_px, poi_rowcol, shape, scale=scale,
                           holes=holes))
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
