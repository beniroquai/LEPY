"""
Read-only view onto a finished LepyLoop run.

A LepyLoop execution produces a folder called ``<input>_Lepy_Execution_<date>``
with this layout (only the parts the reviewer needs are listed)::

    <run>/
        jpgs/<code>.jpg                 copy of the processed source image
        visualisations_png/<code>.png   multi-panel QC figure
        contours_txt/<code>.txt         YOLO-seg polygon, normalised to [0, 1]
        stats_json/<code>.json          full stats dict, POIs in original pixels
        stats_combined.csv              TAB separated, one row per code
        morphological_errors/<code>.png  implausible morphological traits
        morphological_errors_overview.txt
        errors_log/errors*.log          images that failed outright

:class:`LepyRun` indexes such a folder; :class:`SpecimenRecord` gives lazy
access to one specimen.  Nothing here writes -- see
:mod:`napari_lepy.review.export` for that.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
import typing as T
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from napari_lepy.review.palette import POI_NAMES

logger = logging.getLogger(__name__)


#: Sub-folders of a LepyLoop execution folder we care about.
DIR_IMAGES = "jpgs"
DIR_VIS = "visualisations_png"
DIR_CONTOURS = "contours_txt"
DIR_STATS_JSON = "stats_json"
DIR_MORPH_ERRORS = "morphological_errors"
DIR_ERRORS_LOG = "errors_log"

FILE_STATS_COMBINED = "stats_combined.csv"
FILE_MORPH_OVERVIEW = "morphological_errors_overview.txt"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG")

#: ``stats_combined.csv`` is written by ``OutputWriter.write_csv_row``.
STATS_DELIMITER = "\t"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_float(value: T.Any) -> T.Optional[float]:
    """Parse a CSV cell into a float, tolerating '', 'nan' and None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if np.isfinite(val) else None


def _to_int(value: T.Any) -> T.Optional[int]:
    val = _to_float(value)
    return None if val is None else int(round(val))


def read_contour_txt(path: Path, width: int, height: int) -> np.ndarray:
    """Read a YOLO-segmentation polygon file and return pixel coordinates.

    The file holds one line per contour, ``<class> x1 y1 x2 y2 ...`` with all
    coordinates normalised by the image width / height.  Only the first (=
    largest, as written by ``Plotter.save_contour``) contour is returned.

    Returns
    -------
    (N, 2) int32 array in LEPY's ``(x, y)`` convention.
    """
    text = path.read_text().strip()
    if not text:
        return np.zeros((0, 2), dtype=np.int32)
    line = text.splitlines()[0].split()
    coords = np.asarray(line[1:], dtype=float)  # drop the class id
    if coords.size < 6 or coords.size % 2:
        return np.zeros((0, 2), dtype=np.int32)
    coords = coords.reshape(-1, 2)
    coords[:, 0] *= width
    coords[:, 1] *= height
    return np.round(coords).astype(np.int32)


#: Class id of hole polygons in a corrected ``contours_txt`` file.  The outer
#: outline stays the first line (class 0), so readers that only look at the
#: first line -- including :func:`read_contour_txt` -- are unaffected.
HOLE_CLASS_ID = 1


def read_holes_txt(path: Path, width: int, height: int) -> T.List[np.ndarray]:
    """Hole polygons (lines with :data:`HOLE_CLASS_ID`) as ``(N, 2) (x, y)``."""
    holes: T.List[np.ndarray] = []
    for line in path.read_text().strip().splitlines()[1:]:
        parts = line.split()
        if not parts or parts[0] != str(HOLE_CLASS_ID):
            continue
        coords = np.asarray(parts[1:], dtype=float)
        if coords.size < 6 or coords.size % 2:
            continue
        coords = coords.reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        holes.append(np.round(coords).astype(np.int32))
    return holes


def contour_to_txt(contour_px: np.ndarray, width: int, height: int,
                   class_id: int = 0) -> str:
    """Inverse of :func:`read_contour_txt` -- one line, no trailing newline."""
    if contour_px is None or len(contour_px) == 0:
        return ""
    coords = np.asarray(contour_px, dtype=float).reshape(-1, 2).copy()
    coords[:, 0] /= float(width)
    coords[:, 1] /= float(height)
    coords = np.clip(coords, 0.0, 1.0)
    parts = [str(class_id)]
    for x, y in coords:
        parts.append(f"{x:.7f}")
        parts.append(f"{y:.7f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# one specimen
# ---------------------------------------------------------------------------
@dataclass
class SpecimenRecord:
    """Lazy accessor for a single specimen of a :class:`LepyRun`."""

    code: str
    run: "LepyRun"

    # ---- paths --------------------------------------------------------
    @property
    def image_path(self) -> T.Optional[Path]:
        return self.run.image_paths.get(self.code)

    @property
    def vis_path(self) -> T.Optional[Path]:
        path = self.run.root / DIR_VIS / f"{self.code}.png"
        return path if path.exists() else None

    @property
    def contour_path(self) -> T.Optional[Path]:
        path = self.run.root / DIR_CONTOURS / f"{self.code}.txt"
        return path if path.exists() else None

    @property
    def json_path(self) -> T.Optional[Path]:
        path = self.run.root / DIR_STATS_JSON / f"{self.code}.json"
        return path if path.exists() else None

    # ---- stats --------------------------------------------------------
    @property
    def stats(self) -> T.Dict[str, str]:
        """Raw ``stats_combined.csv`` row (strings), ``{}`` if the code failed."""
        return self.run.stats_by_code.get(self.code, {})

    @property
    def has_stats(self) -> bool:
        return self.code in self.run.stats_by_code

    @property
    def failed(self) -> bool:
        """LepyLoop produced no stats for this image."""
        return not self.has_stats

    @property
    def morphology_flagged(self) -> bool:
        return self.code in self.run.morph_error_codes

    @property
    def morphology_reason(self) -> str:
        return self.run.morph_error_reasons.get(self.code, "")

    @property
    def failure_reason(self) -> str:
        return self.run.failure_reasons.get(self.code, "")

    @property
    def scale(self) -> T.Optional[float]:
        """Calibration in px/mm, or ``None`` when calibration failed."""
        val = _to_float(self.stats.get("calibration_length"))
        return val if val and val > 0 else None

    @property
    def size(self) -> T.Optional[T.Tuple[int, int]]:
        """``(height, width)`` of the original image, from the stats row."""
        w = _to_int(self.stats.get("poi_orig_width")) or _to_int(
            self.stats.get("image_width"))
        h = _to_int(self.stats.get("poi_orig_height")) or _to_int(
            self.stats.get("image_height"))
        if w and h:
            return h, w
        return None

    # ---- data ---------------------------------------------------------
    def image(self) -> T.Optional[np.ndarray]:
        """Decode the source image (RGB uint8).  Cached by the run."""
        path = self.image_path
        if path is None:
            return None
        return self.run.cache.get(path)

    def image_shape(self) -> T.Optional[T.Tuple[int, int]]:
        """``(height, width)`` -- from the stats row if possible, else decode."""
        size = self.size
        if size is not None:
            return size
        im = self.image()
        return None if im is None else im.shape[:2]

    def contour_px(self) -> np.ndarray:
        """LEPY contour in original pixels, ``(N, 2)`` as ``(x, y)``."""
        path = self.contour_path
        shape = self.image_shape()
        if path is None or shape is None:
            return np.zeros((0, 2), dtype=np.int32)
        height, width = shape
        try:
            return read_contour_txt(path, width=width, height=height)
        except Exception:
            logger.exception("Could not read contour for %s", self.code)
            return np.zeros((0, 2), dtype=np.int32)

    def poi_rowcol(self) -> T.Optional[np.ndarray]:
        """POIs as an ``(8, 2)`` float array of ``(row, col)``, POI_NAMES order.

        ``None`` when the stats row has no (complete) POI block.
        """
        stats = self.stats
        rows = []
        for name in POI_NAMES:
            x = _to_float(stats.get(f"poi_{name}_x"))
            y = _to_float(stats.get(f"poi_{name}_y"))
            if x is None or y is None:
                return None
            rows.append([y, x])  # napari wants (row, col) == (y, x)
        return np.asarray(rows, dtype=float)

    def full_stats(self) -> T.Dict[str, T.Any]:
        """Typed stats dict -- prefers ``stats_json`` (real numbers) over CSV."""
        path = self.json_path
        if path is not None:
            try:
                with open(path) as handle:
                    return json.load(handle)
            except Exception:
                logger.exception("Could not read stats json for %s", self.code)
        # fall back to the CSV row, converting what we can
        out: T.Dict[str, T.Any] = {}
        for key, value in self.stats.items():
            if key == "Code":
                continue
            num = _to_float(value)
            out[key] = value if num is None else num
        return out


# ---------------------------------------------------------------------------
# image cache
# ---------------------------------------------------------------------------
class ImageCache:
    """Tiny thread-safe LRU cache of decoded images.

    6000x4000 RGB is ~72 MB per image, so the default of four entries keeps the
    current specimen plus the prefetched neighbours resident without growing
    without bound over a 1500-image session.
    """

    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._data: "OrderedDict[Path, np.ndarray]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, path: Path) -> T.Optional[np.ndarray]:
        with self._lock:
            if path in self._data:
                self._data.move_to_end(path)
                return self._data[path]
        image = self._read(path)
        if image is None:
            return None
        with self._lock:
            self._data[path] = image
            self._data.move_to_end(path)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return image

    def peek(self, path: Path) -> bool:
        with self._lock:
            return path in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    @staticmethod
    def _read(path: Path) -> T.Optional[np.ndarray]:
        from skimage.io import imread
        try:
            image = imread(str(path))
        except Exception:
            logger.exception("Could not read image %s", path)
            return None
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
@dataclass
class LepyRun:
    """Index of one LepyLoop execution folder."""

    root: Path
    images_dir: T.Optional[Path] = None
    codes: T.List[str] = field(default_factory=list)
    header: T.List[str] = field(default_factory=list)
    stats_by_code: T.Dict[str, T.Dict[str, str]] = field(default_factory=dict)
    image_paths: T.Dict[str, Path] = field(default_factory=dict)
    morph_error_codes: T.Set[str] = field(default_factory=set)
    morph_error_reasons: T.Dict[str, str] = field(default_factory=dict)
    failure_reasons: T.Dict[str, str] = field(default_factory=dict)
    cache: ImageCache = field(default_factory=ImageCache)

    # ---- construction --------------------------------------------------
    @classmethod
    def discover(cls, path: T.Union[str, Path],
                 images_dir: T.Union[str, Path, None] = None) -> "LepyRun":
        """Build a run index from *path*.

        *path* may be the execution folder itself, or the folder that was fed
        to LepyLoop -- in that case the newest ``*_Lepy_Execution_*`` sibling
        is used.
        """
        path = Path(path).expanduser().resolve()
        root = cls._resolve_run_root(path)
        run = cls(root=root)
        run.images_dir = Path(images_dir).expanduser().resolve() if images_dir \
            else run._guess_images_dir(path)
        run.reload()
        return run

    @staticmethod
    def _resolve_run_root(path: Path) -> Path:
        if (path / FILE_STATS_COMBINED).exists() or (path / DIR_STATS_JSON).is_dir():
            return path
        candidates = sorted(
            (p for p in path.glob("*_Lepy_Execution_*") if p.is_dir()),
            key=lambda p: p.name)
        if candidates:
            return candidates[-1]
        candidates = sorted(
            (p for p in path.parent.glob(f"{path.name}_Lepy_Execution_*")
             if p.is_dir()),
            key=lambda p: p.name)
        if candidates:
            return candidates[-1]
        return path

    def _guess_images_dir(self, opened: Path) -> T.Optional[Path]:
        """Where the full-resolution source images live."""
        internal = self.root / DIR_IMAGES
        if internal.is_dir() and any(internal.iterdir()):
            return internal
        if opened != self.root and opened.is_dir():
            return opened
        # ``<input>_Lepy_Execution_<date>`` -> ``<input>``
        match = re.match(r"^(.*)_Lepy_Execution_.*$", self.root.name)
        if match:
            sibling = self.root.parent / match.group(1)
            if sibling.is_dir():
                return sibling
        return None

    # ---- indexing ------------------------------------------------------
    def reload(self) -> None:
        self._read_stats_combined()
        self._index_images()
        self._read_morph_errors()
        self._read_error_logs()

        codes = set(self.stats_by_code) | set(self.image_paths)
        for sub in (DIR_VIS, DIR_CONTOURS, DIR_STATS_JSON):
            directory = self.root / sub
            if directory.is_dir():
                codes |= {p.stem for p in directory.iterdir() if p.is_file()
                          and not p.name.startswith(".")}
        self.codes = sorted(codes)

    def _read_stats_combined(self) -> None:
        self.stats_by_code, self.header = {}, []
        path = self.root / FILE_STATS_COMBINED
        if not path.exists():
            path = self._concat_per_run_csv()
            if path is None:
                return
        try:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle, delimiter=STATS_DELIMITER)
                self.header = list(reader.fieldnames or [])
                for row in reader:
                    code = (row.get("Code") or "").strip()
                    if code:
                        self.stats_by_code[code] = row
        except Exception:
            logger.exception("Could not read %s", path)

    def _concat_per_run_csv(self) -> T.Optional[Path]:
        """Fall back to ``stats_per_run_csv/`` when the combined file is absent."""
        directory = self.root / "stats_per_run_csv"
        if not directory.is_dir():
            return None
        merged = self.root / FILE_STATS_COMBINED
        rows, header = [], None
        for shard in sorted(directory.glob("*.csv")):
            with open(shard, newline="") as handle:
                reader = csv.reader(handle, delimiter=STATS_DELIMITER)
                shard_header = next(reader, None)
                if shard_header is None:
                    continue
                header = header or shard_header
                rows.extend(row for row in reader if row)
        if header is None:
            return None
        with open(merged, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter=STATS_DELIMITER)
            writer.writerow(header)
            writer.writerows(rows)
        logger.info("Rebuilt %s from %s", merged.name, directory.name)
        return merged

    def _index_images(self) -> None:
        self.image_paths = {}
        for directory in filter(None, (self.root / DIR_IMAGES, self.images_dir)):
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in IMAGE_SUFFIXES:
                    # first hit wins -> the run's own copy is preferred
                    self.image_paths.setdefault(path.stem, path)

    def _read_morph_errors(self) -> None:
        self.morph_error_codes, self.morph_error_reasons = set(), {}
        directory = self.root / DIR_MORPH_ERRORS
        if directory.is_dir():
            self.morph_error_codes = {
                p.stem for p in directory.iterdir()
                if p.is_file() and not p.name.startswith(".")}
        overview = self.root / FILE_MORPH_OVERVIEW
        if not overview.exists():
            return
        try:
            for line in overview.read_text().splitlines():
                if ":" not in line:
                    continue
                reason, _, rest = line.partition(":")
                for code in re.split(r"[,\s]+", rest.strip()):
                    if code:
                        self.morph_error_codes.add(code)
                        previous = self.morph_error_reasons.get(code, "")
                        self.morph_error_reasons[code] = (
                            f"{previous}, {reason.strip()}" if previous
                            else reason.strip())
        except Exception:
            logger.exception("Could not read %s", overview)

    def _read_error_logs(self) -> None:
        self.failure_reasons = {}
        directory = self.root / DIR_ERRORS_LOG
        logs = list(directory.glob("*.log")) if directory.is_dir() else []
        errors_txt = self.root / "errors.log"
        if errors_txt.exists():
            logs.append(errors_txt)
        pattern = re.compile(r'Failed to process "(?P<path>[^"]+)"\.\s*Reason\s*'
                             r'\((?P<kind>[^)]*)\):\s*(?P<msg>.*)')
        for log in logs:
            try:
                for line in log.read_text(errors="replace").splitlines():
                    match = pattern.search(line)
                    if match:
                        code = Path(match.group("path")).stem
                        self.failure_reasons[code] = \
                            f"{match.group('kind')}: {match.group('msg')}".strip()
            except Exception:
                logger.exception("Could not read %s", log)

    # ---- access --------------------------------------------------------
    def record(self, code: str) -> SpecimenRecord:
        return SpecimenRecord(code=code, run=self)

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def default_output_dir(self) -> Path:
        """Sibling folder the corrections are written to by default."""
        match = re.match(r"^(?P<stem>.*)_Lepy_Execution_(?P<stamp>.*)$",
                         self.root.name)
        if match:
            return self.root.parent / (
                f"{match.group('stem')}_Napari_Corrections_{match.group('stamp')}")
        return self.root.parent / f"{self.root.name}_Napari_Corrections"

    def batches(self, size: int) -> T.List[T.List[str]]:
        size = max(1, int(size))
        return [self.codes[i:i + size] for i in range(0, len(self.codes), size)]

    def summary(self) -> str:
        n_vis = sum(1 for c in self.codes if (self.root / DIR_VIS / f"{c}.png").exists())
        return (f"{len(self.codes)} specimens | {len(self.stats_by_code)} with stats | "
                f"{n_vis} QC figures | {len(self.morph_error_codes)} morphology-flagged | "
                f"{len(self.codes) - len(self.stats_by_code)} failed")
