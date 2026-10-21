"""
LEPY Review & Correction -- step through a finished LepyLoop run, confirm the
automatic results, and fix the ones that are wrong without leaving napari.

Workflow
--------
1. **Open** a ``*_Lepy_Execution_*`` folder.  Everything else (source images,
   QC figures, contours, stats, error logs) is discovered from it.
2. **Step** through the specimens with the arrow keys.  Each one shows the
   full-resolution image with the LEPY contour and the eight POIs on top, plus
   the pre-rendered QC figure for a quick verdict.
3. **Judge**: ``V`` confirms the LEPY values, ``N`` flags the specimen as
   wrong.  Both are recorded in ``review_status.csv``.
4. **Correct**: drag POIs, or edit the outline (polygon vertices or a brush).
   Any edit marks the specimen ``corrected``; saving recomputes the affected
   statistics with the original LEPY code and writes them to the output folder.
5. **Export** hands LepyLoop a list of corrected codes so only that subset has
   to be re-run.
"""
from __future__ import annotations

import logging
import typing as T
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QTextEdit, QMessageBox, QScrollArea, QListWidget, QListWidgetItem,
    QProgressBar, QSizePolicy, QDialog, QFrame, QApplication,
)

import napari

from napari_lepy.review import palette as pal
from napari_lepy.review import recompute as rc
from napari_lepy.review import status as st
from napari_lepy.review.dataset import LepyRun, SpecimenRecord
from napari_lepy.review.export import CorrectionWriter
from napari_lepy.review.status import ReviewStatusDB

logger = logging.getLogger(__name__)


FILTERS: T.List[T.Tuple[str, T.Optional[T.Callable[["LEPYReviewWidget", str], bool]]]] = [
    ("All specimens", None),
    ("Not reviewed", lambda w, c: w.db.status(c) == st.UNREVIEWED),
    ("Needs correction", lambda w, c: w.db.status(c) == st.FLAGGED),
    ("Corrected", lambda w, c: w.db.status(c) == st.CORRECTED),
    ("Confirmed OK", lambda w, c: w.db.status(c) == st.OK),
    ("Morphology-flagged", lambda w, c: w.run.record(c).morphology_flagged),
    ("LEPY failed", lambda w, c: w.run.record(c).failed),
]

# Shortcut letters must not collide with napari's own *layer* keymaps, because
# the active layer is consulted before the viewer.  Taken as of napari 0.8:
#   Points  A P S Z                    Labels  B E F L M P X Z - = [ ]
#   Shapes  A B D E F I L P R S T X Z  Image   C O X Y Z
# ...which rules out the obvious O / X / T.  G H J K N Q U V W are free.
KEY_CONFIRM, KEY_FLAG, KEY_SAVE = "V", "N", "W"
KEY_QC, KEY_RESET, KEY_UNREVIEW = "Q", "G", "U"

#: Contour-layer styling: the outline and the holes cut out of it.
OUTLINE_EDGE, OUTLINE_FACE = "#00FF88", "#00FF8820"
HOLE_EDGE, HOLE_FACE = "#FF3355", "#FF335540"
#: Smallest hole kept, as a fraction of the image area (~50x50 px at 24 MP).
MIN_HOLE_FRAC = 1e-4

KEY_HELP = (
    f"<b>←/→</b> prev/next &nbsp;·&nbsp; <b>{KEY_CONFIRM}</b> confirm correct "
    f"&nbsp;·&nbsp; <b>{KEY_FLAG}</b> needs correction &nbsp;·&nbsp; "
    f"<b>{KEY_SAVE}</b> save correction &nbsp;·&nbsp; <b>{KEY_QC}</b> QC figure "
    f"&nbsp;·&nbsp; <b>{KEY_RESET}</b> reset to LEPY &nbsp;·&nbsp; "
    f"<b>{KEY_UNREVIEW}</b> un-review"
)


class LEPYReviewWidget(QWidget):
    """Dock widget implementing the batch review / correction loop."""

    L_IMAGE = "Specimen"
    L_QC = "QC figure"
    L_CONTOUR = "Contour"
    L_MASK = "Mask (brush)"
    L_POI = "POI"

    OWNED_LAYERS = (L_IMAGE, L_QC, L_CONTOUR, L_MASK, L_POI)

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.run: T.Optional[LepyRun] = None
        self.db: T.Optional[ReviewStatusDB] = None
        self.writer: T.Optional[CorrectionWriter] = None
        self.output_dir: T.Optional[Path] = None

        #: code -> corrected stats dict (mirrors ``<out>/stats_json``)
        self.corrected_stats: T.Dict[str, T.Dict[str, T.Any]] = {}

        self._view_codes: T.List[str] = []
        self._position = 0
        self._record: T.Optional[SpecimenRecord] = None
        self._orig_poi: T.Optional[np.ndarray] = None
        self._orig_contour: T.Optional[np.ndarray] = None
        self._orig_holes: T.List[np.ndarray] = []
        self._shape: T.Optional[T.Tuple[int, int]] = None
        self._loading = False
        self._poi_dirty = False
        self._contour_dirty = False
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="lepy-prefetch")
        self._prefetch_jobs: T.List[T.Any] = []

        self._build_ui()
        self._bind_keys()
        self._set_enabled(False)

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(self._build_data_box())
        layout.addWidget(self._build_navigation_box())
        layout.addWidget(self._build_verdict_box())
        layout.addWidget(self._build_qc_box())
        layout.addWidget(self._build_correction_box())
        layout.addWidget(self._build_export_box())
        layout.addStretch(1)

    # ---- ① data --------------------------------------------------------
    def _build_data_box(self) -> QGroupBox:
        box = QGroupBox("① LEPY run")
        v = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_open = QPushButton("Open LEPY run folder…")
        self.btn_open.clicked.connect(self._on_open_run)
        row.addWidget(self.btn_open)
        v.addLayout(row)

        self.lbl_run = QLabel("No run loaded.")
        self.lbl_run.setWordWrap(True)
        self.lbl_run.setStyleSheet("font-size: 11px;")
        v.addWidget(self.lbl_run)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output:"))
        self.lbl_output = QLabel("—")
        self.lbl_output.setWordWrap(True)
        self.lbl_output.setStyleSheet("font-size: 11px; color: #888;")
        row.addWidget(self.lbl_output, 1)
        self.btn_output = QPushButton("Change…")
        self.btn_output.setMaximumWidth(80)
        self.btn_output.clicked.connect(self._on_change_output)
        row.addWidget(self.btn_output)
        v.addLayout(row)
        return box

    # ---- ② navigation --------------------------------------------------
    def _build_navigation_box(self) -> QGroupBox:
        box = QGroupBox("② Batch & navigation")
        v = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Batch size:"))
        self.sp_batch_size = QSpinBox()
        self.sp_batch_size.setRange(1, 100000)
        self.sp_batch_size.setValue(100)
        self.sp_batch_size.valueChanged.connect(lambda _: self._refresh_view(reset=True))
        row.addWidget(self.sp_batch_size)
        row.addStretch(1)
        self.btn_batch_prev = QPushButton("◀")
        self.btn_batch_prev.setMaximumWidth(34)
        self.btn_batch_prev.clicked.connect(lambda: self._step_batch(-1))
        row.addWidget(self.btn_batch_prev)
        self.lbl_batch = QLabel("batch —")
        self.lbl_batch.setAlignment(Qt.AlignCenter)
        self.lbl_batch.setMinimumWidth(90)
        row.addWidget(self.lbl_batch)
        self.btn_batch_next = QPushButton("▶")
        self.btn_batch_next.setMaximumWidth(34)
        self.btn_batch_next.clicked.connect(lambda: self._step_batch(1))
        row.addWidget(self.btn_batch_next)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Show:"))
        self.cb_filter = QComboBox()
        self.cb_filter.addItems([name for name, _ in FILTERS])
        self.cb_filter.currentIndexChanged.connect(lambda _: self._refresh_view())
        row.addWidget(self.cb_filter, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        row.addWidget(self.btn_prev)
        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.clicked.connect(lambda: self._step(1))
        row.addWidget(self.btn_next)
        v.addLayout(row)

        self.list_codes = QListWidget()
        self.list_codes.setMinimumHeight(150)
        self.list_codes.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.list_codes.currentRowChanged.connect(self._on_list_row_changed)
        v.addWidget(self.list_codes)

        self.bar_progress = QProgressBar()
        self.bar_progress.setFormat("%v / %m reviewed")
        v.addWidget(self.bar_progress)

        hint = QLabel(KEY_HELP)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        v.addWidget(hint)
        return box

    # ---- ③ verdict -----------------------------------------------------
    def _build_verdict_box(self) -> QGroupBox:
        box = QGroupBox("③ Verdict")
        v = QVBoxLayout(box)

        self.lbl_code = QLabel("—")
        self.lbl_code.setStyleSheet("font-size: 14px; font-weight: bold;")
        v.addWidget(self.lbl_code)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(
            "padding: 3px; border-radius: 3px; color: white; background: #888;")
        v.addWidget(self.lbl_status)

        row = QHBoxLayout()
        self.btn_ok = QPushButton(f"✓  Correct  ({KEY_CONFIRM})")
        self.btn_ok.setStyleSheet(
            "background: #2E9E4F; color: white; font-weight: bold; padding: 6px;")
        self.btn_ok.clicked.connect(lambda: self._mark(st.OK, advance=True))
        row.addWidget(self.btn_ok)
        self.btn_flag = QPushButton(f"✗  Needs fix  ({KEY_FLAG})")
        self.btn_flag.setStyleSheet(
            "background: #CC3311; color: white; font-weight: bold; padding: 6px;")
        self.btn_flag.clicked.connect(lambda: self._mark(st.FLAGGED, advance=False))
        row.addWidget(self.btn_flag)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_skip = QPushButton("»  Skip")
        self.btn_skip.clicked.connect(lambda: self._mark(st.SKIPPED, advance=True))
        row.addWidget(self.btn_skip)
        self.btn_unreview = QPushButton(f"↺  Un-review  ({KEY_UNREVIEW})")
        self.btn_unreview.clicked.connect(self._on_unreview)
        row.addWidget(self.btn_unreview)
        v.addLayout(row)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.setStyleSheet("font-size: 11px; color: #CC7700;")
        v.addWidget(self.lbl_notice)
        return box

    # ---- ④ QC figure ---------------------------------------------------
    def _build_qc_box(self) -> QGroupBox:
        box = QGroupBox("④ QC figure")
        v = QVBoxLayout(box)

        row = QHBoxLayout()
        self.cb_triage = QCheckBox(f"Show QC figure in canvas  ({KEY_QC})")
        self.cb_triage.toggled.connect(self._on_triage_toggled)
        row.addWidget(self.cb_triage)
        v.addLayout(row)

        self.lbl_qc = QLabel("—")
        self.lbl_qc.setAlignment(Qt.AlignCenter)
        self.lbl_qc.setMinimumHeight(120)
        self.lbl_qc.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.lbl_qc.setStyleSheet("border: 1px solid #444;")
        v.addWidget(self.lbl_qc)

        self.btn_qc_full = QPushButton("Open QC figure full size…")
        self.btn_qc_full.clicked.connect(self._on_qc_fullsize)
        v.addWidget(self.btn_qc_full)
        return box

    # ---- ⑤ correction --------------------------------------------------
    def _build_correction_box(self) -> QGroupBox:
        box = QGroupBox("⑤ Correction")
        v = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colours:"))
        self.cb_scheme = QComboBox()
        for key in pal.SCHEMES:
            self.cb_scheme.addItem(pal.SCHEME_LABELS[key], key)
        self.cb_scheme.setCurrentIndex(
            list(pal.SCHEMES).index(pal.DEFAULT_SCHEME))
        self.cb_scheme.currentIndexChanged.connect(lambda _: self._restyle_poi())
        row.addWidget(self.cb_scheme, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Marker size %:"))
        self.sp_poi_size = QDoubleSpinBox()
        self.sp_poi_size.setRange(0.1, 5.0)
        self.sp_poi_size.setSingleStep(0.1)
        self.sp_poi_size.setValue(0.9)
        self.sp_poi_size.valueChanged.connect(lambda _: self._restyle_poi())
        row.addWidget(self.sp_poi_size)
        self.cb_labels = QCheckBox("labels")
        self.cb_labels.setChecked(True)
        self.cb_labels.toggled.connect(lambda _: self._restyle_poi())
        row.addWidget(self.cb_labels)
        v.addLayout(row)

        v.addWidget(self._build_legend())

        hint = QLabel(
            "🖱️  Select the <b>POI</b> layer, press <kbd>3</kbd> (select mode) "
            "and drag a point. Select the <b>Contour</b> layer and press "
            "<kbd>4</kbd> to move outline vertices. Red polygons in the "
            "Contour layer are <b>holes</b> cut out of the mask.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        v.addWidget(hint)

        row = QHBoxLayout()
        self.btn_simplify = QPushButton("Simplify outline")
        self.btn_simplify.setToolTip(
            "Reduce the number of outline vertices so they can be dragged "
            "comfortably. Changes the contour, so it counts as a correction.")
        self.btn_simplify.clicked.connect(self._on_simplify)
        row.addWidget(self.btn_simplify)
        self.sp_epsilon = QSpinBox()
        self.sp_epsilon.setRange(1, 200)
        self.sp_epsilon.setValue(6)
        self.sp_epsilon.setSuffix(" px")
        row.addWidget(self.sp_epsilon)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_to_brush = QPushButton("Outline → brush mask")
        self.btn_to_brush.setToolTip(
            "Rasterise the outline into a Labels layer so it can be painted "
            "with the brush (P) and eraser (E).")
        self.btn_to_brush.clicked.connect(self._on_to_brush)
        row.addWidget(self.btn_to_brush)
        self.btn_from_brush = QPushButton("Brush mask → outline")
        self.btn_from_brush.clicked.connect(self._on_from_brush)
        row.addWidget(self.btn_from_brush)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_draw = QPushButton("Draw new outline")
        self.btn_draw.setToolTip(
            "Start an empty polygon layer — use this for specimens LEPY could "
            "not segment at all.")
        self.btn_draw.clicked.connect(self._on_draw_outline)
        row.addWidget(self.btn_draw)
        self.btn_paint = QPushButton("Paint new mask")
        self.btn_paint.setToolTip(
            "Start an empty Labels layer and paint the animal with the brush.")
        self.btn_paint.clicked.connect(self._on_paint_mask)
        row.addWidget(self.btn_paint)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_cut_hole = QPushButton("Cut hole")
        self.btn_cut_hole.setToolTip(
            "Draw a polygon around background that the outline wrongly "
            "encloses (e.g. an antenna or leg touching the body twice). It is "
            "removed from the mask, the areas and the colour statistics. "
            "Delete a hole by selecting it (S) and pressing Backspace.")
        self.btn_cut_hole.clicked.connect(self._on_cut_hole)
        row.addWidget(self.btn_cut_hole)
        self.btn_find_holes = QPushButton("Find background holes")
        self.btn_find_holes.setToolTip(
            "Suggest holes: regions inside the outline whose colour matches "
            "the background just outside it. Check the suggestions -- pale "
            "wing patches can look like background.")
        self.btn_find_holes.clicked.connect(self._on_find_holes)
        row.addWidget(self.btn_find_holes)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_reset = QPushButton(f"↺  Reset to LEPY  ({KEY_RESET})")
        self.btn_reset.clicked.connect(self._on_reset_to_lepy)
        row.addWidget(self.btn_reset)
        self.btn_redetect = QPushButton("Re-detect POI from outline")
        self.btn_redetect.setToolTip(
            "Run lepy's automatic POI detection again on the current outline.")
        self.btn_redetect.clicked.connect(self._on_redetect_poi)
        row.addWidget(self.btn_redetect)
        v.addLayout(row)

        self.cb_recolor = QCheckBox(
            "Recompute colour statistics when the outline changed (slow)")
        self.cb_recolor.setChecked(True)
        v.addWidget(self.cb_recolor)

        self.cb_autosave = QCheckBox("Auto-save edits when moving on")
        self.cb_autosave.setChecked(True)
        v.addWidget(self.cb_autosave)

        self.btn_save = QPushButton(f"💾  Save correction  ({KEY_SAVE})")
        self.btn_save.setStyleSheet(
            "background: #0077BB; color: white; font-weight: bold; padding: 6px;")
        self.btn_save.clicked.connect(self._on_save)
        v.addWidget(self.btn_save)

        self.txt_traits = QTextEdit()
        self.txt_traits.setReadOnly(True)
        self.txt_traits.setMaximumHeight(150)
        self.txt_traits.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self.txt_traits)
        return box

    def _build_legend(self) -> QWidget:
        frame = QFrame()
        grid = QGridLayout(frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        self._legend_buttons: T.List[QPushButton] = []
        for index, name in enumerate(pal.POI_NAMES):
            button = QPushButton(pal.POI_LABELS[name])
            button.setFlat(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(f"{name} — click to select and centre on it")
            button.clicked.connect(lambda _=False, i=index: self._select_poi(i))
            self._legend_buttons.append(button)
            grid.addWidget(button, index // 2, index % 2)
        self._restyle_legend()
        return frame

    # ---- ⑥ export ------------------------------------------------------
    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("⑥ Export")
        v = QVBoxLayout(box)
        self.lbl_counts = QLabel("—")
        self.lbl_counts.setWordWrap(True)
        self.lbl_counts.setStyleSheet("font-size: 11px;")
        v.addWidget(self.lbl_counts)
        row = QHBoxLayout()
        self.btn_export = QPushButton("Write tables & code lists")
        self.btn_export.clicked.connect(self._on_export)
        row.addWidget(self.btn_export)
        self.btn_reveal = QPushButton("Open folder")
        self.btn_reveal.clicked.connect(self._on_reveal)
        row.addWidget(self.btn_reveal)
        v.addLayout(row)
        return box

    # =====================================================================
    # keyboard
    # =====================================================================
    def _bind_keys(self) -> None:
        bindings = {
            "Right": lambda _v: self._step(1),
            "Left": lambda _v: self._step(-1),
            KEY_CONFIRM: lambda _v: self._mark(st.OK, advance=True),
            KEY_FLAG: lambda _v: self._mark(st.FLAGGED, advance=False),
            KEY_SAVE: lambda _v: self._on_save(),
            KEY_QC: lambda _v: self.cb_triage.setChecked(
                not self.cb_triage.isChecked()),
            KEY_RESET: lambda _v: self._on_reset_to_lepy(),
            KEY_UNREVIEW: lambda _v: self._on_unreview(),
        }
        for key, callback in bindings.items():
            try:
                self.viewer.bind_key(key, callback, overwrite=True)
            except Exception:
                logger.warning("Could not bind key %s", key)

    # =====================================================================
    # ① opening a run
    # =====================================================================
    def _on_open_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a LEPY execution folder (…_Lepy_Execution_…)")
        if not folder:
            return
        self.open_run(Path(folder))

    def open_run(self, folder: Path,
                 output_dir: T.Optional[Path] = None) -> None:
        """Index *folder* and start a review session (also usable from scripts)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.run = LepyRun.discover(folder)
            self.output_dir = Path(output_dir) if output_dir \
                else self.run.default_output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.db = ReviewStatusDB(
                self.output_dir / ReviewStatusDB.FILENAME,
                source_run=self.run.name)
            self.writer = CorrectionWriter(self.output_dir, self.run, self.db)
            self.corrected_stats = self.writer.load_corrected_stats()
            self._seed_status()
        except Exception as exc:
            logger.exception("Could not open run")
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not open run", str(exc))
            return
        QApplication.restoreOverrideCursor()

        images = self.run.images_dir or (self.run.root / "jpgs")
        self.lbl_run.setText(
            f"<b>{self.run.name}</b><br>{self.run.summary()}"
            f"<br><span style='color:#888'>images: {images}</span>")
        self.lbl_output.setText(str(self.output_dir))
        self._set_enabled(True)
        self._refresh_view(reset=True)

    def _seed_status(self) -> None:
        """Give every code a row, and pre-mark the ones LEPY could not process."""
        assert self.run is not None and self.db is not None
        changed = False
        for code in self.run.codes:
            entry = self.db.get(code)
            if code not in self.run.stats_by_code and entry.status == st.UNREVIEWED:
                entry.status = st.FAILED
                entry.note = entry.note or self.run.failure_reasons.get(code, "")
                changed = True
        if changed or not self.db.path.exists():
            self.db.save()

    def _on_change_output(self) -> None:
        if self.run is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose an output folder for corrections",
            str(self.output_dir or self.run.root.parent))
        if folder:
            self.open_run(self.run.root, Path(folder))

    # =====================================================================
    # ② navigation
    # =====================================================================
    @property
    def batch_size(self) -> int:
        return int(self.sp_batch_size.value())

    @property
    def batch_index(self) -> int:
        return getattr(self, "_batch_index", 0)

    def _batch_codes(self) -> T.List[str]:
        if self.run is None:
            return []
        batches = self.run.batches(self.batch_size)
        if not batches:
            return []
        self._batch_index = max(0, min(self.batch_index, len(batches) - 1))
        return batches[self._batch_index]

    def _refresh_view(self, reset: bool = False) -> None:
        """Recompute the visible code list from batch + filter, refresh the UI."""
        if self.run is None:
            return
        self._flush_pending()
        if reset:
            self._batch_index = 0
        codes = self._batch_codes()
        _, predicate = FILTERS[self.cb_filter.currentIndex()]
        if predicate is not None:
            codes = [c for c in codes if predicate(self, c)]

        previous = self._record.code if self._record else None
        previous_position = self._position
        self._view_codes = codes

        batches = self.run.batches(self.batch_size)
        self.lbl_batch.setText(
            f"batch {self._batch_index + 1}/{max(1, len(batches))}")
        self._update_counts()

        if not codes:
            self._rebuild_list()
            self._clear_layers()
            self.lbl_code.setText("— no specimen matches this filter —")
            self.lbl_status.setText("")
            self._set_specimen_enabled(False)
            return

        if previous in codes:
            self._position = codes.index(previous)
        else:
            # The current specimen dropped out of the filter (usually because it
            # was just reviewed) -- holding the index lands on the next one.
            self._position = min(previous_position, len(codes) - 1)
        self._rebuild_list()
        self._load_current()

    def _rebuild_list(self) -> None:
        self.list_codes.blockSignals(True)
        self.list_codes.clear()
        for code in self._view_codes:
            status = self.db.status(code)
            item = QListWidgetItem(f"{st.STATUS_ICON.get(status, '·')}  {code}")
            item.setForeground(QColor(st.STATUS_COLOR.get(status, "#888888")))
            record = self.run.record(code)
            tips = [st.STATUS_LABEL.get(status, status)]
            if record.morphology_flagged:
                tips.append(f"morphology: {record.morphology_reason or 'flagged'}")
            if record.failed and record.failure_reason:
                tips.append(record.failure_reason)
            item.setToolTip("\n".join(tips))
            self.list_codes.addItem(item)
        self.list_codes.setCurrentRow(
            min(self._position, self.list_codes.count() - 1))
        self.list_codes.blockSignals(False)

    def _on_list_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._view_codes) or row == self._position:
            return
        self._position = row
        self._load_current()

    def _step(self, delta: int) -> None:
        if not self._view_codes:
            return
        target = self._position + delta
        if target < 0 or target >= len(self._view_codes):
            self._step_batch(1 if delta > 0 else -1, edge=True)
            return
        self._position = target
        self.list_codes.blockSignals(True)
        self.list_codes.setCurrentRow(self._position)
        self.list_codes.blockSignals(False)
        self._load_current()

    def _step_batch(self, delta: int, edge: bool = False) -> None:
        if self.run is None:
            return
        batches = self.run.batches(self.batch_size)
        target = self.batch_index + delta
        if target < 0 or target >= len(batches):
            self.lbl_notice.setText(
                "Start of the dataset." if delta < 0 else "End of the dataset.")
            return
        self._flush_pending()          # before dropping the current record
        self._batch_index = target
        self._record = None            # so _refresh_view lands on the batch edge
        self._position = 0
        self._refresh_view()
        if edge and self._view_codes:
            self._position = 0 if delta > 0 else len(self._view_codes) - 1
            self.list_codes.setCurrentRow(self._position)
            self._load_current()

    # =====================================================================
    # loading a specimen
    # =====================================================================
    def _load_current(self) -> None:
        if not self._view_codes:
            return
        self._flush_pending()
        code = self._view_codes[self._position]
        self._load_specimen(code)
        self._prefetch()

    def _load_specimen(self, code: str) -> None:
        assert self.run is not None
        record = self.run.record(code)
        self._record = record
        self._poi_dirty = self._contour_dirty = False
        self.lbl_notice.setText("")

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._loading = True
        try:
            image = record.image()
            shape = record.image_shape() or (
                image.shape[:2] if image is not None else None)
            self._shape = shape

            contour, holes = self._contour_for(record)
            poi = self._poi_for(record)
            self._orig_contour = record.contour_px()
            self._orig_holes = []
            self._orig_poi = record.poi_rowcol()

            self._update_image_layer(image)
            self._update_qc_layer(record, shape)
            self._update_contour_layer(contour, holes)
            self._update_poi_layer(poi)
            self._drop_layer(self.L_MASK)
            self._apply_triage(self.cb_triage.isChecked())
            if image is not None or record.vis_path is not None:
                self.viewer.reset_view()
        finally:
            self._loading = False
            QApplication.restoreOverrideCursor()

        self._update_qc_thumbnail(record)
        self._update_specimen_info()
        self._show_traits(self._current_stats())

    def _contour_for(self, record: SpecimenRecord
                     ) -> T.Tuple[np.ndarray, T.List[np.ndarray]]:
        """Corrected outline and holes if saved earlier, else the LEPY one."""
        if self.writer is not None:
            path = self.writer.corrected_contour_path(record.code)
            shape = record.image_shape()
            if path is not None and shape is not None:
                from napari_lepy.review.dataset import (
                    read_contour_txt, read_holes_txt)
                try:
                    return (read_contour_txt(path, width=shape[1], height=shape[0]),
                            read_holes_txt(path, width=shape[1], height=shape[0]))
                except Exception:
                    logger.exception("Could not read corrected contour for %s",
                                     record.code)
        return record.contour_px(), []

    def _poi_for(self, record: SpecimenRecord) -> T.Optional[np.ndarray]:
        stats = self.corrected_stats.get(record.code)
        if stats:
            poi = rc.rowcol_from_stats(stats)
            if poi is not None:
                return poi
        return record.poi_rowcol()

    def _current_stats(self) -> T.Dict[str, T.Any]:
        if self._record is None:
            return {}
        return dict(self.corrected_stats.get(self._record.code)
                    or self._record.full_stats())

    # ---- layers --------------------------------------------------------
    def _update_image_layer(self, image: T.Optional[np.ndarray]) -> None:
        if image is None:
            self._drop_layer(self.L_IMAGE)
            return
        if self.L_IMAGE in self.viewer.layers:
            self.viewer.layers[self.L_IMAGE].data = image
        else:
            self.viewer.add_image(image, name=self.L_IMAGE, rgb=True)
        self.viewer.layers[self.L_IMAGE].visible = not self.cb_triage.isChecked()

    def _update_qc_layer(self, record: SpecimenRecord,
                         shape: T.Optional[T.Tuple[int, int]]) -> None:
        path = record.vis_path
        if path is None:
            self._drop_layer(self.L_QC)
            self._qc_pixmap = None
            return
        from skimage.io import imread
        try:
            figure = imread(str(path))
        except Exception:
            logger.exception("Could not read QC figure %s", path)
            self._drop_layer(self.L_QC)
            return
        if figure.ndim == 3 and figure.shape[-1] == 4:
            figure = figure[..., :3]

        # Fit the QC figure into the specimen's world extent, undistorted, so
        # that reset_view frames both the same way and toggling between them
        # does not move the camera.
        scale, translate = (1.0, 1.0), (0.0, 0.0)
        if shape is not None:
            height, width = shape
            factor = min(height / figure.shape[0], width / figure.shape[1])
            scale = (factor, factor)
            translate = ((height - figure.shape[0] * factor) / 2,
                         (width - figure.shape[1] * factor) / 2)

        if self.L_QC in self.viewer.layers:
            layer = self.viewer.layers[self.L_QC]
            layer.data = figure
            layer.scale = scale
            layer.translate = translate
        else:
            self.viewer.add_image(figure, name=self.L_QC, rgb=True,
                                  scale=scale, translate=translate,
                                  visible=self.cb_triage.isChecked())

    def _update_contour_layer(self, contour: np.ndarray,
                              holes: T.Optional[T.Sequence[np.ndarray]] = None
                              ) -> None:
        """Outline first, then every hole as its own (red) polygon."""
        if contour is None or len(contour) < 3:
            self._drop_layer(self.L_CONTOUR)
            return
        holes = [h for h in holes or [] if len(h) >= 3]
        polygons = [rc.contour_to_polygon(c) for c in [contour, *holes]]
        width = self._line_width()
        if self.L_CONTOUR not in self.viewer.layers:
            layer = self.viewer.add_shapes(
                name=self.L_CONTOUR, edge_width=width, opacity=0.9)
            layer.events.data.connect(self._on_contour_edited)
        layer = self.viewer.layers[self.L_CONTOUR]
        layer.data = []
        layer.add(polygons, shape_type="polygon", edge_width=width,
                  edge_color=[OUTLINE_EDGE] + [HOLE_EDGE] * len(holes),
                  face_color=[OUTLINE_FACE] + [HOLE_FACE] * len(holes))
        layer.current_edge_color = OUTLINE_EDGE
        layer.current_face_color = OUTLINE_FACE
        layer.visible = not self.cb_triage.isChecked()

    def _update_poi_layer(self, poi: T.Optional[np.ndarray]) -> None:
        if poi is None:
            self._drop_layer(self.L_POI)
            return
        features = {"poi": np.array([pal.POI_SHORT[n] for n in pal.POI_NAMES])}
        if self.L_POI in self.viewer.layers:
            layer = self.viewer.layers[self.L_POI]
            layer.data = poi
            layer.features = features
        else:
            layer = self.viewer.add_points(
                poi, name=self.L_POI, features=features,
                border_color="white", border_width=0.12)
            layer.events.data.connect(self._on_poi_edited)
        self.viewer.layers[self.L_POI].visible = not self.cb_triage.isChecked()
        self._restyle_poi()

    def _restyle_poi(self) -> None:
        if self.L_POI not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.L_POI]
        scheme = self.cb_scheme.currentData() or pal.DEFAULT_SCHEME
        colors = pal.colors(scheme)
        symbols = pal.symbols(scheme)
        size = self._marker_size()
        was_loading, self._loading = self._loading, True
        try:
            layer.face_color = colors
            layer.size = size
            try:
                layer.symbol = symbols
            except Exception:
                layer.symbol = symbols[0]
            layer.text = {
                "string": "{poi}",
                # Per-point colour, not white: the specimen background is a
                # pale grey sheet, on which white labels are barely readable.
                "color": colors,
                "size": 10,
                "translation": [-size * 0.9, 0],
                "visible": self.cb_labels.isChecked(),
            }
        except Exception:
            logger.exception("Could not restyle the POI layer")
        finally:
            self._loading = was_loading
        self._restyle_legend()

    def _restyle_legend(self) -> None:
        scheme = self.cb_scheme.currentData() or pal.DEFAULT_SCHEME
        styles = pal.styles(scheme)
        marks = {"disc": "●", "square": "■", "diamond": "◆",
                 "triangle_up": "▲", "triangle_down": "▼"}
        for button, name in zip(getattr(self, "_legend_buttons", []),
                                pal.POI_NAMES):
            style = styles[name]
            button.setText(f"{marks.get(style.symbol, '●')} {pal.POI_LABELS[name]}")
            button.setStyleSheet(
                f"color: {style.color}; font-weight: bold; font-size: 11px; "
                "text-align: left; border: none;")

    def _marker_size(self) -> float:
        if not self._shape:
            return 12.0
        return max(4.0, self.sp_poi_size.value() / 100.0 * max(self._shape))

    def _line_width(self) -> float:
        if not self._shape:
            return 2.0
        return max(1.0, 0.0008 * max(self._shape))

    def _select_poi(self, index: int) -> None:
        if self.L_POI not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.L_POI]
        data = np.asarray(layer.data)
        if index >= len(data):
            return
        self.viewer.layers.selection.active = layer
        layer.mode = "select"
        layer.selected_data = {index}
        self.viewer.camera.center = (0, float(data[index, 0]), float(data[index, 1]))

    def _drop_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

    def _clear_layers(self) -> None:
        for name in self.OWNED_LAYERS:
            self._drop_layer(name)

    # ---- triage --------------------------------------------------------
    def _on_triage_toggled(self, checked: bool) -> None:
        self._apply_triage(checked)

    def _apply_triage(self, checked: bool) -> None:
        for name in (self.L_IMAGE, self.L_CONTOUR, self.L_POI, self.L_MASK):
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = not checked
        if self.L_QC in self.viewer.layers:
            self.viewer.layers[self.L_QC].visible = checked
        elif checked:
            self.lbl_notice.setText("No QC figure exists for this specimen.")

    def _update_qc_thumbnail(self, record: SpecimenRecord) -> None:
        path = record.vis_path
        if path is None:
            self.lbl_qc.setText("no QC figure")
            self.lbl_qc.setPixmap(QPixmap())
            self._qc_path = None
            return
        self._qc_path = path
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.lbl_qc.setText("QC figure could not be read")
            return
        width = max(200, self.lbl_qc.width() - 4)
        self.lbl_qc.setPixmap(pixmap.scaledToWidth(
            width, Qt.SmoothTransformation))
        self.lbl_qc.setFixedHeight(
            int(width * pixmap.height() / max(1, pixmap.width())))

    def _on_qc_fullsize(self) -> None:
        path = getattr(self, "_qc_path", None)
        if path is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(str(Path(path).name))
        layout = QVBoxLayout(dialog)
        area = QScrollArea()
        area.setWidgetResizable(True)
        label = QLabel()
        label.setPixmap(QPixmap(str(path)))
        label.setScaledContents(False)
        area.setWidget(label)
        layout.addWidget(area)
        screen = QApplication.primaryScreen().availableGeometry()
        dialog.resize(QSize(int(screen.width() * 0.9), int(screen.height() * 0.9)))
        dialog.exec_()

    # =====================================================================
    # ③ verdicts
    # =====================================================================
    def _mark(self, status: str, advance: bool) -> None:
        if self._record is None or self.db is None:
            return
        if status == st.OK and (self._poi_dirty or self._contour_dirty):
            answer = QMessageBox.question(
                self, "Unsaved edits",
                "This specimen has unsaved edits. Save them as a correction "
                "instead of confirming the LEPY values?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Save:
                self._on_save()
                if advance:
                    self._step(1)
                return
            self._poi_dirty = self._contour_dirty = False
        self.db.set_status(self._record.code, status, batch=self.batch_index)
        self.db.save()
        self._refresh_after_status_change(advance)

    def _on_unreview(self) -> None:
        if self._record is None or self.db is None:
            return
        self.db.reset(self._record.code)
        self.db.save()
        self._refresh_after_status_change(advance=False)

    def _refresh_after_status_change(self, advance: bool) -> None:
        _, predicate = FILTERS[self.cb_filter.currentIndex()]
        if predicate is not None:
            # the current specimen may have dropped out of the filter
            self._refresh_view()
            return
        self._rebuild_list()
        self._update_counts()
        self._update_specimen_info()
        if advance:
            self._step(1)

    # =====================================================================
    # ⑤ corrections
    # =====================================================================
    def _on_poi_edited(self, event=None) -> None:
        if self._loading or self._record is None:
            return
        self._poi_dirty = True
        self._update_specimen_info()

    def _on_contour_edited(self, event=None) -> None:
        if self._loading or self._record is None:
            return
        self._contour_dirty = True
        self._update_specimen_info()

    def _current_poi(self) -> T.Optional[np.ndarray]:
        if self.L_POI not in self.viewer.layers:
            return None
        data = np.asarray(self.viewer.layers[self.L_POI].data, dtype=float)
        return data if data.shape == (len(pal.POI_NAMES), 2) else data

    def _current_polygons(self) -> T.List[np.ndarray]:
        """Contour-layer polygons as ``(x, y)`` contours, largest area first."""
        if self.L_CONTOUR not in self.viewer.layers:
            return []
        contours = [rc.polygon_to_contour(p)
                    for p in self.viewer.layers[self.L_CONTOUR].data]
        contours = [c for c in contours if len(c) >= 3]
        return sorted(contours, key=lambda c: -abs(cv2.contourArea(
            c.reshape(-1, 1, 2).astype(np.float32))))

    def _current_contour(self) -> np.ndarray:
        """The outer outline: the largest polygon of the Contour layer."""
        polygons = self._current_polygons()
        return polygons[0] if polygons else np.zeros((0, 2), dtype=np.int32)

    def _current_holes(self) -> T.List[np.ndarray]:
        """Every further polygon of the Contour layer is a hole."""
        return self._current_polygons()[1:]

    def _on_simplify(self) -> None:
        contour = self._current_contour()
        if len(contour) < 4:
            return
        reduced = rc.simplify_contour(contour, float(self.sp_epsilon.value()))
        self._loading = True
        try:
            self._update_contour_layer(reduced, self._current_holes())
        finally:
            self._loading = False
        self._contour_dirty = True
        self.lbl_notice.setText(
            f"Outline simplified: {len(contour)} → {len(reduced)} vertices. "
            "Saving will store the simplified outline.")
        self._update_specimen_info()

    def _on_draw_outline(self) -> None:
        """Empty polygon layer -- the entry point for specimens LEPY skipped."""
        self._drop_layer(self.L_CONTOUR)
        layer = self.viewer.add_shapes(
            name=self.L_CONTOUR, shape_type="polygon",
            edge_color="#00FF88", face_color="#00FF8820",
            edge_width=self._line_width(), opacity=0.9)
        layer.events.data.connect(self._on_contour_edited)
        layer.mode = "add_polygon"
        self.viewer.layers.selection.active = layer
        self._contour_dirty = True
        self.lbl_notice.setText(
            "Click to place outline vertices, press <Escape> to finish, then "
            "“Re-detect POI from outline”.")

    def _on_paint_mask(self) -> None:
        """Empty Labels layer for painting a mask from scratch."""
        if self._shape is None:
            return
        self._drop_layer(self.L_MASK)
        layer = self.viewer.add_labels(
            np.zeros(self._shape, dtype=np.uint8), name=self.L_MASK, opacity=0.4)
        layer.selected_label = 1
        layer.mode = "paint"
        layer.brush_size = max(10, int(0.01 * max(self._shape)))
        self.viewer.layers.selection.active = layer
        if self.L_CONTOUR in self.viewer.layers:
            self.viewer.layers[self.L_CONTOUR].visible = False
        self.lbl_notice.setText(
            "Paint the animal with the brush (P), then press "
            "“Brush mask → outline”.")

    def _on_to_brush(self) -> None:
        contour = self._current_contour()
        if self._shape is None or len(contour) < 3:
            return
        mask = rc.rasterize_contour(contour, self._shape, self._current_holes())
        if self.L_MASK in self.viewer.layers:
            self.viewer.layers[self.L_MASK].data = mask
        else:
            self.viewer.add_labels(mask.astype(np.uint8), name=self.L_MASK,
                                   opacity=0.4)
        layer = self.viewer.layers[self.L_MASK]
        layer.visible = not self.cb_triage.isChecked()
        layer.selected_label = 1
        self.viewer.layers.selection.active = layer
        if self.L_CONTOUR in self.viewer.layers:
            self.viewer.layers[self.L_CONTOUR].visible = False
        self.lbl_notice.setText(
            "Paint with the brush (P) / eraser (E), then press "
            "“Brush mask → outline”.")

    def _on_from_brush(self) -> None:
        if self.L_MASK not in self.viewer.layers:
            self.lbl_notice.setText("No brush mask layer — create one first.")
            return
        mask = np.asarray(self.viewer.layers[self.L_MASK].data)
        contour = rc.contour_from_mask(mask)
        if len(contour) < 3:
            self._warn("Empty mask", "The brush mask contains no object.")
            return
        # Erased patches inside the animal become holes; brush speckles don't.
        holes = rc.holes_from_mask(
            mask, min_area_px=MIN_HOLE_FRAC * mask.shape[0] * mask.shape[1])
        self._loading = True
        try:
            self._update_contour_layer(contour, holes)
        finally:
            self._loading = False
        self.viewer.layers[self.L_CONTOUR].visible = not self.cb_triage.isChecked()
        self._contour_dirty = True
        self.lbl_notice.setText(
            f"Outline rebuilt from the brush mask ({len(contour)} vertices, "
            f"{len(holes)} hole{'' if len(holes) == 1 else 's'}).")
        self._update_specimen_info()

    def _on_cut_hole(self) -> None:
        """Switch the Contour layer to polygon drawing for a new hole."""
        if self.L_CONTOUR not in self.viewer.layers:
            self.lbl_notice.setText("No outline — draw or paint one first.")
            return
        layer = self.viewer.layers[self.L_CONTOUR]
        layer.visible = True
        layer.current_edge_color = HOLE_EDGE
        layer.current_face_color = HOLE_FACE
        layer.mode = "add_polygon"
        self.viewer.layers.selection.active = layer
        self.lbl_notice.setText(
            "Click around the background patch, press <Escape> to finish. "
            "Every polygon besides the outline is cut out of the mask.")

    def _on_find_holes(self) -> None:
        record = self._record
        contour = self._current_contour()
        if record is None or len(contour) < 3:
            return
        image = record.image()
        if image is None:
            self.lbl_notice.setText("No image — cannot look for holes.")
            return
        holes = self._current_holes()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = rc.detect_background_holes(
                image, contour, holes, min_area_frac=MIN_HOLE_FRAC)
        finally:
            QApplication.restoreOverrideCursor()
        if not found:
            self.lbl_notice.setText("No background enclosed by the outline found.")
            return
        self._loading = True
        try:
            self._update_contour_layer(contour, holes + found)
        finally:
            self._loading = False
        self._contour_dirty = True
        self.lbl_notice.setText(
            f"{len(found)} hole{'' if len(found) == 1 else 's'} suggested (red). "
            "Delete wrong ones: select with S, then Backspace.")
        self._update_specimen_info()

    def _on_redetect_poi(self) -> None:
        contour = self._current_contour()
        if self._shape is None or len(contour) < 3:
            return
        from lepy.poi import PointsOfInterest
        filled = rc.rasterize_contour(contour, self._shape, self._current_holes())
        try:
            pois = PointsOfInterest.detect(filled)
        except Exception as exc:
            logger.exception("POI detection failed")
            self._warn("POI detection failed",
                       f"{exc}\n\nFix the outline first, or move the points "
                       "manually.")
            return
        poi = np.asarray([[getattr(pois, n).row, getattr(pois, n).col]
                          for n in pal.POI_NAMES], dtype=float)
        self._loading = True
        try:
            self._update_poi_layer(poi)
        finally:
            self._loading = False
        self._poi_dirty = True
        self._update_specimen_info()

    def _on_reset_to_lepy(self) -> None:
        if self._record is None:
            return
        self._loading = True
        try:
            self._update_contour_layer(self._record.contour_px(), [])
            self._update_poi_layer(self._record.poi_rowcol())
            self._drop_layer(self.L_MASK)
        finally:
            self._loading = False
        self._poi_dirty = self._contour_dirty = False
        self.lbl_notice.setText("Reset to the original LEPY values.")
        self._update_specimen_info()

    # ---- saving --------------------------------------------------------
    def _flush_pending(self) -> None:
        if (self._poi_dirty or self._contour_dirty) and self.cb_autosave.isChecked():
            self._on_save(quiet=True)

    def _on_save(self, quiet: bool = False) -> None:
        if self._record is None or self.writer is None or self._shape is None:
            return
        record = self._record
        poi = self._current_poi()
        contour = self._current_contour()

        if poi is None or poi.shape != (len(pal.POI_NAMES), 2):
            self._warn(
                "Wrong number of POIs",
                f"The POI layer must hold exactly {len(pal.POI_NAMES)} points "
                f"in the original order, but it holds "
                f"{0 if poi is None else len(poi)}.\n\n"
                "Use “Reset to LEPY” or “Re-detect POI from outline” to "
                "restore them.")
            return
        if len(contour) < 3:
            self._warn("No outline", "The contour layer holds no usable polygon.")
            return

        holes = self._current_holes()
        mask_changed = (self._contour_dirty
                        or not _same_contour(contour, self._orig_contour)
                        or len(holes) != len(self._orig_holes)
                        or not all(_same_contour(a, b)
                                   for a, b in zip(holes, self._orig_holes)))
        moved, max_shift = rc.poi_shift(self._orig_poi, poi)
        poi_changed = self._poi_dirty or moved > 0

        if not (mask_changed or poi_changed):
            if not quiet:
                self.lbl_notice.setText("Nothing changed — nothing to save.")
            return

        # A painted mask may legitimately have holes; a filled polygon cannot.
        # Use whichever the user actually edited for the colour histograms.
        brush_mask = None
        if self.L_MASK in self.viewer.layers:
            brush_mask = (np.asarray(self.viewer.layers[self.L_MASK].data) > 0
                          ).astype(np.uint8)

        previous = self.db.get(record.code)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            stats = rc.recompute(
                record.full_stats(),
                contour_px=contour, poi_rowcol=poi, shape=self._shape,
                scale=record.scale, mask_changed=mask_changed,
                rgb=record.image() if mask_changed else None,
                color_mask=brush_mask,
                recompute_color=self.cb_recolor.isChecked(),
                holes=holes)
            mask = None
            if mask_changed:
                mask = brush_mask if brush_mask is not None \
                    else rc.rasterize_contour(contour, self._shape, holes)
            self.writer.write_specimen(record.code, stats,
                                       contour_px=contour, shape=self._shape,
                                       mask=mask, holes=holes)
            self.corrected_stats[record.code] = stats
            # Flags accumulate: a later POI-only save must not clear the fact
            # that the outline was edited in an earlier one.
            self.db.set_status(
                record.code, st.CORRECTED, batch=self.batch_index,
                poi_corrected=bool(poi_changed or previous.poi_corrected),
                mask_corrected=bool(mask_changed or previous.mask_corrected),
                needs_reanalysis=bool(mask_changed or previous.needs_reanalysis),
                n_poi_moved=int(max(moved, previous.n_poi_moved)),
                max_poi_shift_px=float(max(
                    previous.max_poi_shift_px,
                    0.0 if max_shift == float("inf") else max_shift)))
            self.writer.write_tables(self.corrected_stats)
        except Exception as exc:
            logger.exception("Saving the correction failed")
            QApplication.restoreOverrideCursor()
            self._warn("Could not save the correction", str(exc))
            return
        QApplication.restoreOverrideCursor()

        self._poi_dirty = self._contour_dirty = False
        self._orig_poi, self._orig_contour = poi.copy(), contour.copy()
        self._orig_holes = [h.copy() for h in holes]
        self._show_traits(stats)
        self._rebuild_list()
        self._update_counts()
        self._update_specimen_info()
        if not quiet:
            what = []
            if poi_changed:
                what.append(f"{moved} POI moved (max {max_shift:.0f} px)")
            if mask_changed:
                what.append("outline changed → flagged for re-analysis")
                if holes:
                    what.append(f"{len(holes)} hole(s) cut out")
            self.lbl_notice.setText("Saved: " + ", ".join(what))

    # =====================================================================
    # info panels
    # =====================================================================
    def _update_specimen_info(self) -> None:
        if self._record is None or self.db is None:
            return
        record = self._record
        status = self.db.status(record.code)
        self.lbl_code.setText(
            f"{record.code}  <span style='font-size:11px;color:#888'>"
            f"({self._position + 1}/{len(self._view_codes)})</span>")

        text = st.STATUS_LABEL.get(status, status)
        if self._poi_dirty or self._contour_dirty:
            text += "  ·  unsaved edits"
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            "padding: 3px; border-radius: 3px; color: white; background: "
            f"{'#CC7700' if (self._poi_dirty or self._contour_dirty) else st.STATUS_COLOR.get(status, '#888888')};")

        notices = []
        if record.failed:
            notices.append(
                "LEPY produced no result for this image"
                + (f": {record.failure_reason}" if record.failure_reason else "")
                + ". Draw an outline and re-detect the POIs, or leave it flagged "
                  "for re-analysis.")
        if record.morphology_flagged:
            notices.append(
                f"Flagged by LEPY's plausibility check: {record.morphology_reason}")
        if record.image_path is None:
            notices.append("The source image could not be found.")
        if record.scale is None and not record.failed:
            notices.append("No calibration — distances stay in pixels.")
        if notices:
            self.lbl_notice.setText(" ".join(notices))

        self._set_specimen_enabled(True)
        editable = record.image_path is not None
        for button in (self.btn_save, self.btn_simplify, self.btn_to_brush,
                       self.btn_from_brush, self.btn_redetect, self.btn_reset,
                       self.btn_draw, self.btn_paint, self.btn_cut_hole,
                       self.btn_find_holes):
            button.setEnabled(editable)

        # Look the row up by code -- ``_position`` may already point at the next
        # specimen when this runs from an auto-save during navigation.
        if record.code in self._view_codes:
            row = self._view_codes.index(record.code)
            if 0 <= row < self.list_codes.count():
                item = self.list_codes.item(row)
                item.setText(f"{st.STATUS_ICON.get(status, '·')}  {record.code}")
                item.setForeground(QColor(st.STATUS_COLOR.get(status, "#888888")))

    def _show_traits(self, stats: T.Mapping[str, T.Any]) -> None:
        from lepy.outputs import OUTPUTS as OUTS
        if not stats:
            self.txt_traits.setPlainText("(no statistics)")
            return
        scale = self._record.scale if self._record else None
        unit = "mm" if scale else "px"
        area_unit = "mm²" if scale else "px²"

        # LEPY truncates POI coordinates to whole calibration units before
        # measuring, so show the un-quantised value next to it.
        poi = self._current_poi()
        exact: T.Dict[str, float] = {}
        if poi is not None and poi.shape == (len(pal.POI_NAMES), 2):
            try:
                exact = rc.exact_distances(poi, scale=scale)
            except Exception:
                exact = {}

        rows: T.List[str] = []
        if scale:
            rows.append(f"  {'Calibration':<17s} {scale:>9.2f}  px/mm")
        rows.append(f"  {'':<17s} {'LEPY':>9s}  {'exact':>9s}")
        for key, label in (
            (OUTS.poi.dist.inner_outer_l, "Left FWL"),
            (OUTS.poi.dist.inner_outer_r, "Right FWL"),
            (OUTS.poi.dist.inner, "Body width"),
            (OUTS.poi.dist.body, "Body length"),
        ):
            value = _as_float(stats.get(key))
            if value is None:
                continue
            other = exact.get(f"{key}{rc.EXACT_SUFFIX}",
                              _as_float(stats.get(f"{key}{rc.EXACT_SUFFIX}")))
            tail = f"{other:>9.2f}" if other is not None else " " * 9
            rows.append(f"  {label:<17s} {value:>9.2f}  {tail}  {unit}")

        rows.append("")
        for key, label, suffix in (
            (OUTS.contour.width_calibrated, "Wing span", unit),
            (OUTS.poi.area.wing_l, "Left wing area", area_unit),
            (OUTS.poi.area.wing_r, "Right wing area", area_unit),
            (OUTS.poi.area.body, "Body area", area_unit),
            (OUTS.contour.area, "Contour area", "px²"),
            (OUTS.contour.length, "Contour vertices", ""),
        ):
            value = _as_float(stats.get(key))
            if value is None:
                continue
            rows.append(f"  {label:<17s} {value:>9.2f}  {' ' * 9}  {suffix}")
        self.txt_traits.setPlainText("\n".join(rows))

    def _update_counts(self) -> None:
        if self.run is None or self.db is None:
            return
        counts = self.db.counts(self.run.codes)
        done, total = self.db.progress(self.run.codes)
        self.bar_progress.setMaximum(max(1, total))
        self.bar_progress.setValue(done)
        self.lbl_counts.setText(
            f"✓ {counts.get(st.OK, 0)} confirmed &nbsp; "
            f"✎ {counts.get(st.CORRECTED, 0)} corrected &nbsp; "
            f"✗ {counts.get(st.FLAGGED, 0)} needs fix &nbsp; "
            f"» {counts.get(st.SKIPPED, 0)} skipped &nbsp; "
            f"! {counts.get(st.FAILED, 0)} failed &nbsp; "
            f"· {counts.get(st.UNREVIEWED, 0)} open")

    # =====================================================================
    # ⑥ export
    # =====================================================================
    def _on_export(self) -> None:
        if self.writer is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            written = self.writer.write_tables(self.corrected_stats)
        except Exception as exc:
            logger.exception("Export failed")
            QApplication.restoreOverrideCursor()
            self._warn("Export failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        QMessageBox.information(
            self, "Export complete",
            f"Written to:\n{self.output_dir}\n\n"
            + "\n".join(f"· {p.name}" for p in written)
            + f"\n\n{len(self.db.corrected_codes())} corrected specimens, "
              f"{len(self.db.reanalysis_codes())} need a LepyLoop re-run.")

    def _on_reveal(self) -> None:
        if self.output_dir is not None:
            webbrowser.open(self.output_dir.as_uri())

    # =====================================================================
    # prefetch & housekeeping
    # =====================================================================
    def _prefetch(self) -> None:
        """Decode the neighbouring images in the background.

        Outstanding jobs are cancelled first: holding the arrow key down would
        otherwise queue a 72 MB decode for every specimen it swept past.
        """
        if self.run is None:
            return
        for future in self._prefetch_jobs:
            future.cancel()
        self._prefetch_jobs = []
        for offset in (1, -1, 2):
            index = self._position + offset
            if not 0 <= index < len(self._view_codes):
                continue
            path = self.run.image_paths.get(self._view_codes[index])
            if path is not None and not self.run.cache.peek(path):
                self._prefetch_jobs.append(
                    self._pool.submit(self.run.cache.get, path))

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.sp_batch_size, self.cb_filter, self.list_codes,
                       self.btn_batch_prev, self.btn_batch_next,
                       self.btn_export, self.btn_reveal, self.btn_output):
            widget.setEnabled(enabled)
        self._set_specimen_enabled(enabled)

    def _set_specimen_enabled(self, enabled: bool) -> None:
        for widget in (self.btn_prev, self.btn_next, self.btn_ok, self.btn_flag,
                       self.btn_skip, self.btn_unreview, self.btn_save,
                       self.btn_simplify, self.btn_to_brush, self.btn_from_brush,
                       self.btn_reset, self.btn_redetect, self.btn_qc_full,
                       self.btn_draw, self.btn_paint, self.btn_cut_hole,
                       self.btn_find_holes):
            widget.setEnabled(enabled)

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event):  # noqa: N802  (Qt naming)
        try:
            self._flush_pending()
        finally:
            self._pool.shutdown(wait=False)
        super().closeEvent(event)


def _as_float(value: T.Any) -> T.Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_contour(a: T.Optional[np.ndarray], b: T.Optional[np.ndarray]) -> bool:
    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))
