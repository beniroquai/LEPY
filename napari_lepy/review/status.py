"""
The review database: which specimens have been checked, and what happened.

Backed by a single comma-separated ``review_status.csv`` in the correction
output folder -- small enough (1 row per specimen) to rewrite atomically on
every change, and readable by LepyLoop / Excel / pandas without extra tooling.

The column that matters downstream is ``needs_reanalysis``: it is ``True``
whenever the *mask* changed, because then colour statistics and the contour
have to be recomputed from the full-resolution image.  A pure POI move does not
require a re-run -- those numbers are already recomputed here.
"""
from __future__ import annotations

import csv
import datetime as dt
import getpass
import logging
import os
import tempfile
import typing as T
from dataclasses import dataclass, asdict, fields
from pathlib import Path

logger = logging.getLogger(__name__)


# --- status values ----------------------------------------------------------
UNREVIEWED = "unreviewed"
OK = "ok"                # LEPY values confirmed correct, nothing changed
CORRECTED = "corrected"  # user edited POIs and/or the mask, and saved
FLAGGED = "flagged"      # marked wrong, but not corrected here
SKIPPED = "skipped"      # deliberately passed over
FAILED = "failed"        # LepyLoop produced no result for this image

ALL_STATUSES = (UNREVIEWED, OK, CORRECTED, FLAGGED, SKIPPED, FAILED)

STATUS_ICON = {
    UNREVIEWED: "·",
    OK: "✓",
    CORRECTED: "✎",
    FLAGGED: "✗",
    SKIPPED: "»",
    FAILED: "!",
}

STATUS_COLOR = {
    UNREVIEWED: "#888888",
    OK: "#2E9E4F",
    CORRECTED: "#0077BB",
    FLAGGED: "#CC3311",
    SKIPPED: "#AA8800",
    FAILED: "#AA4499",
}

STATUS_LABEL = {
    UNREVIEWED: "not reviewed",
    OK: "confirmed correct",
    CORRECTED: "corrected",
    FLAGGED: "needs correction",
    SKIPPED: "skipped",
    FAILED: "LEPY failed",
}


@dataclass
class ReviewEntry:
    """One row of ``review_status.csv``."""

    code: str
    status: str = UNREVIEWED
    poi_corrected: bool = False
    mask_corrected: bool = False
    n_poi_moved: int = 0
    max_poi_shift_px: float = 0.0
    needs_reanalysis: bool = False
    reviewer: str = ""
    reviewed_at: str = ""
    batch: int = -1
    note: str = ""
    source_run: str = ""

    @property
    def reviewed(self) -> bool:
        return self.status not in (UNREVIEWED,)

    def to_row(self) -> T.Dict[str, str]:
        row = asdict(self)
        for key, value in row.items():
            if isinstance(value, bool):
                row[key] = "True" if value else "False"
            elif isinstance(value, float):
                row[key] = f"{value:.2f}"
            else:
                row[key] = "" if value is None else str(value)
        return row

    @classmethod
    def from_row(cls, row: T.Mapping[str, str]) -> "ReviewEntry":
        kwargs: T.Dict[str, T.Any] = {}
        for spec in fields(cls):
            raw = row.get(spec.name)
            if raw is None or raw == "":
                continue
            if spec.type is bool or spec.type == "bool":
                kwargs[spec.name] = str(raw).strip().lower() in ("true", "1", "yes")
            elif spec.type is int or spec.type == "int":
                try:
                    kwargs[spec.name] = int(float(raw))
                except ValueError:
                    pass
            elif spec.type is float or spec.type == "float":
                try:
                    kwargs[spec.name] = float(raw)
                except ValueError:
                    pass
            else:
                kwargs[spec.name] = str(raw)
        kwargs.setdefault("code", str(row.get("code", "")))
        return cls(**kwargs)


COLUMNS = [spec.name for spec in fields(ReviewEntry)]


class ReviewStatusDB:
    """Load / mutate / persist ``review_status.csv``."""

    FILENAME = "review_status.csv"

    def __init__(self, path: Path, source_run: str = ""):
        self.path = Path(path)
        self.source_run = source_run
        self.entries: T.Dict[str, ReviewEntry] = {}
        self.reviewer = self._default_reviewer()
        self.load()

    # ---- persistence ---------------------------------------------------
    @staticmethod
    def _default_reviewer() -> str:
        try:
            return getpass.getuser()
        except Exception:
            return ""

    def load(self) -> None:
        self.entries = {}
        if not self.path.exists():
            return
        try:
            with open(self.path, newline="") as handle:
                for row in csv.DictReader(handle):
                    entry = ReviewEntry.from_row(row)
                    if entry.code:
                        self.entries[entry.code] = entry
        except Exception:
            logger.exception("Could not read %s", self.path)

    def save(self) -> None:
        """Atomic rewrite -- a crash mid-review never truncates the database."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", newline="", delete=False, dir=str(self.path.parent),
            prefix=".review_status_", suffix=".tmp")
        try:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for code in sorted(self.entries):
                writer.writerow(self.entries[code].to_row())
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.path)

    # ---- access --------------------------------------------------------
    def get(self, code: str) -> ReviewEntry:
        entry = self.entries.get(code)
        if entry is None:
            entry = ReviewEntry(code=code, source_run=self.source_run)
            self.entries[code] = entry
        return entry

    def status(self, code: str) -> str:
        entry = self.entries.get(code)
        return entry.status if entry else UNREVIEWED

    def set_status(self, code: str, status: str, *, batch: int = -1,
                   note: T.Optional[str] = None, **extra: T.Any) -> ReviewEntry:
        entry = self.get(code)
        entry.status = status
        entry.reviewer = self.reviewer
        entry.reviewed_at = dt.datetime.now().astimezone().isoformat(
            timespec="seconds")
        entry.source_run = entry.source_run or self.source_run
        if batch >= 0:
            entry.batch = batch
        if note is not None:
            entry.note = note
        for key, value in extra.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        return entry

    def reset(self, code: str) -> ReviewEntry:
        entry = ReviewEntry(code=code, source_run=self.source_run)
        self.entries[code] = entry
        return entry

    # ---- reporting -----------------------------------------------------
    def counts(self, codes: T.Optional[T.Iterable[str]] = None) -> T.Dict[str, int]:
        codes = list(codes) if codes is not None else list(self.entries)
        out = {status: 0 for status in ALL_STATUSES}
        for code in codes:
            out[self.status(code)] = out.get(self.status(code), 0) + 1
        return out

    def corrected_codes(self) -> T.List[str]:
        """Codes with a manual correction -- the subset LepyLoop should re-run."""
        return sorted(code for code, entry in self.entries.items()
                      if entry.status == CORRECTED)

    def reanalysis_codes(self) -> T.List[str]:
        """Codes whose mask changed, so colour/contour stats must be redone."""
        return sorted(code for code, entry in self.entries.items()
                      if entry.needs_reanalysis)

    def flagged_codes(self) -> T.List[str]:
        return sorted(code for code, entry in self.entries.items()
                      if entry.status == FLAGGED)

    def progress(self, codes: T.Iterable[str]) -> T.Tuple[int, int]:
        """``(handled, total)``.

        ``FAILED`` counts as *not* handled: those specimens still need a human
        decision (rescue them, or skip them explicitly), and hiding them behind
        a full progress bar is how they get forgotten.
        """
        codes = list(codes)
        done = sum(1 for code in codes
                   if self.status(code) not in (UNREVIEWED, FAILED))
        return done, len(codes)
