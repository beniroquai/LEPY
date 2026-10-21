"""
Write corrections back out, mirroring the LepyLoop folder layout.

::

    <output>/
        review_status.csv           which specimens were checked, and how
        corrected_codes.txt         one code per line -- the LepyLoop hand-over
        reanalysis_codes.txt        subset whose mask changed
        stats_corrected.csv         TAB separated, corrected rows only
        stats_combined_merged.csv   full original table, corrected rows swapped in
        stats_json/<code>.json      corrected stats, same shape as LepyLoop's
        contours_txt/<code>.txt     corrected polygon, same normalised format;
                                    holes follow as extra lines, class id 1
        masks_png/<code>.png        corrected binary mask (only if it changed)

Original file names (the specimen code) are kept everywhere, so LepyLoop can
join on ``Code`` without any extra bookkeeping.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import typing as T
from pathlib import Path

import numpy as np

from napari_lepy.review.dataset import (
    LepyRun, STATS_DELIMITER, contour_to_txt, HOLE_CLASS_ID,
)
from napari_lepy.review.status import ReviewStatusDB

logger = logging.getLogger(__name__)


DIR_STATS_JSON = "stats_json"
DIR_CONTOURS = "contours_txt"
DIR_MASKS = "masks_png"

FILE_STATS_CORRECTED = "stats_corrected.csv"
FILE_STATS_MERGED = "stats_combined_merged.csv"
FILE_CORRECTED_CODES = "corrected_codes.txt"
FILE_REANALYSIS_CODES = "reanalysis_codes.txt"


def default_header() -> T.List[str]:
    """The ``stats.csv`` column order of :class:`lepy.outputs.writer.OutputWriter`.

    Kept in sync with ``lepy/outputs/writer/file_writer.py``; used only as a
    fallback when the source run has no ``stats_combined.csv`` to copy the
    column order from.
    """
    from lepy.outputs import OUTPUTS as OUTS
    from napari_lepy.review.palette import POI_NAMES

    header = ["Code", OUTS.image.width, OUTS.image.height]
    for stat in (OUTS.intensity, OUTS.saturation, OUTS.hue):
        header += [stat.min, stat.max, stat.median, stat.mean, stat.stddev]
    for stat in (OUTS.red, OUTS.green, OUTS.blue, OUTS.uv, OUTS.rgbuv):
        header += [stat.min, stat.max, stat.median, stat.Q25, stat.Q75,
                   stat.IQR, stat.shannon, stat.simpson]
    for stat in (OUTS.luminance, OUTS.chromaticity_red, OUTS.chromaticity_green,
                 OUTS.chromaticity_blue, OUTS.chromaticity_uv):
        header += [stat.min, stat.max, stat.median, stat.mean, stat.stddev]
    header += [
        OUTS.contour.length, OUTS.contour.area,
        OUTS.contour.xmin, OUTS.contour.xmax, OUTS.contour.ymin, OUTS.contour.ymax,
        OUTS.contour.area_calibrated, OUTS.contour.width_calibrated,
        OUTS.contour.height_calibrated,
        OUTS.calibration.length, OUTS.calibration.score,
        OUTS.calibration.pos.x, OUTS.calibration.pos.y,
        OUTS.calibration.pos.w, OUTS.calibration.pos.h,
        OUTS.poi.area.body, OUTS.poi.area.wing_l, OUTS.poi.area.wing_r,
        OUTS.poi.dist.inner_outer_l, OUTS.poi.dist.inner_outer_r,
        OUTS.poi.dist.inner, OUTS.poi.dist.body,
        OUTS.poi.orig_width, OUTS.poi.orig_height,
        OUTS.poi.center.x, OUTS.poi.center.y,
    ]
    for name in POI_NAMES:
        coord = getattr(OUTS.poi, name)
        header += [coord.x, coord.y]
    return header


def _atomic_write(path: Path, write: T.Callable[[T.IO[str]], None],
                  newline: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline=newline, delete=False, dir=str(path.parent),
        prefix=f".{path.stem}_", suffix=".tmp")
    try:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


class CorrectionWriter:
    """Persist corrected specimens under *out_root*."""

    def __init__(self, out_root: Path, run: LepyRun, db: ReviewStatusDB):
        self.root = Path(out_root)
        self.run = run
        self.db = db
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- header -------------------------------------------------------
    @property
    def header(self) -> T.List[str]:
        """Column order of ``stats_combined.csv``.

        Taken from the source run so the corrected tables line up column for
        column; :func:`default_header` only kicks in when the run has no
        combined table at all.
        """
        return list(self.run.header) if self.run.header else default_header()

    # ---- per specimen --------------------------------------------------
    def write_specimen(self, code: str, stats: T.Mapping[str, T.Any], *,
                       contour_px: T.Optional[np.ndarray] = None,
                       shape: T.Optional[T.Tuple[int, int]] = None,
                       mask: T.Optional[np.ndarray] = None,
                       holes: T.Optional[T.Sequence[np.ndarray]] = None
                       ) -> T.List[Path]:
        """Write the JSON, contour (+ holes) and (optionally) mask for one
        specimen."""
        written: T.List[Path] = []

        json_path = self.root / DIR_STATS_JSON / f"{code}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(json_path,
                      lambda fh: json.dump(_jsonable(stats), fh, indent=2),
                      newline="\n")
        written.append(json_path)

        if contour_px is not None and shape is not None and len(contour_px):
            height, width = shape
            lines = [contour_to_txt(contour_px, width=width, height=height)]
            lines += [contour_to_txt(hole, width=width, height=height,
                                     class_id=HOLE_CLASS_ID)
                      for hole in holes or [] if len(hole) >= 3]
            text = "\n".join(lines)
            contour_path = self.root / DIR_CONTOURS / f"{code}.txt"
            contour_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(contour_path, lambda fh: fh.write(text + "\n"),
                          newline="\n")
            written.append(contour_path)

        if mask is not None:
            from skimage.io import imsave
            mask_path = self.root / DIR_MASKS / f"{code}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            imsave(str(mask_path), ((np.asarray(mask) > 0) * 255).astype(np.uint8),
                   check_contrast=False)
            written.append(mask_path)

        return written

    # ---- tables --------------------------------------------------------
    def write_tables(self, corrected_stats: T.Mapping[str, T.Mapping[str, T.Any]]
                     ) -> T.List[Path]:
        """Rewrite every roll-up table from the current in-memory state."""
        header = self.header
        written: T.List[Path] = []

        def write_corrected(handle: T.IO[str]) -> None:
            writer = csv.writer(handle, delimiter=STATS_DELIMITER)
            writer.writerow(header)
            for code in sorted(corrected_stats):
                writer.writerow(_row(code, corrected_stats[code], header))

        path = self.root / FILE_STATS_CORRECTED
        _atomic_write(path, write_corrected)
        written.append(path)

        def write_merged(handle: T.IO[str]) -> None:
            writer = csv.writer(handle, delimiter=STATS_DELIMITER)
            writer.writerow(header)
            for code in self.run.codes:
                if code in corrected_stats:
                    writer.writerow(_row(code, corrected_stats[code], header))
                elif code in self.run.stats_by_code:
                    row = self.run.stats_by_code[code]
                    writer.writerow([row.get(key, "") for key in header])

        path = self.root / FILE_STATS_MERGED
        _atomic_write(path, write_merged)
        written.append(path)

        for filename, codes in (
            (FILE_CORRECTED_CODES, self.db.corrected_codes()),
            (FILE_REANALYSIS_CODES, self.db.reanalysis_codes()),
        ):
            path = self.root / filename
            _atomic_write(path,
                          lambda fh, c=codes: fh.write("\n".join(c) + ("\n" if c else "")),
                          newline="\n")
            written.append(path)

        self.db.save()
        written.append(self.db.path)
        return written

    # ---- reload --------------------------------------------------------
    def load_corrected_stats(self) -> T.Dict[str, T.Dict[str, T.Any]]:
        """Read back previously corrected specimens so a session can resume."""
        out: T.Dict[str, T.Dict[str, T.Any]] = {}
        directory = self.root / DIR_STATS_JSON
        if not directory.is_dir():
            return out
        for path in sorted(directory.glob("*.json")):
            try:
                with open(path) as handle:
                    out[path.stem] = json.load(handle)
            except Exception:
                logger.exception("Could not read %s", path)
        return out

    def corrected_contour_path(self, code: str) -> T.Optional[Path]:
        path = self.root / DIR_CONTOURS / f"{code}.txt"
        return path if path.exists() else None


# ---------------------------------------------------------------------------
def _jsonable(stats: T.Mapping[str, T.Any]) -> T.Dict[str, T.Any]:
    out: T.Dict[str, T.Any] = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _row(code: str, stats: T.Mapping[str, T.Any],
         header: T.Sequence[str]) -> T.List[T.Any]:
    row: T.List[T.Any] = []
    for key in header:
        if key == "Code":
            row.append(code)
            continue
        value = stats.get(key, "")
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        row.append("" if value is None else value)
    return row
