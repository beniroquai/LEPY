"""
LEPY napari plugin widget.

Architecture
------------
This widget owns one :class:`lepy.Image` instance which is the *single source
of truth* for the pipeline state (``rgb_im``, ``uv_im``, ``mask``, ``contour``,
``stats``, ``key_points``). Each step calls into the original LEPY package:

    ① Load              -> ``Image.read``
    ② Segment           -> ``Image.segment`` (lepy.segmentation.segment)
    ③ Calibrate         -> ``Image.calibrate`` (uses lepy + scalebar)
    ④ Detect POI        -> ``Image.pois`` (lepy.poi.PointsOfInterest.detect)
    ⑤ Colour stats      -> ``Image.color_stats``
    ⑥ Export            -> ``lepy.OutputWriter`` + ``lepy.Plotter``

The plugin only adds:

    * napari layers per intermediate step,
    * a brush workflow on the ``Mask`` Labels layer that lets the user fix
      the U-Net / Otsu / GrabCut output by hand and re-run POI detection,
    * a Points layer the user can drag; on update the new positions become
      ``Image.key_points`` so that ``PointsOfInterest.from_keypoints`` is
      used instead of ``.detect`` (this is an existing extensibility hook
      of LEPY).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import typing as T

import numpy as np
import cv2
from skimage.io import imread, imsave

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QTextEdit,
    QMessageBox, QScrollArea,
)

import napari

# --- Original LEPY package (unmodified) -------------------------------------
from lepy import Image as LepyImage
from lepy import PointsOfInterest
from lepy.outputs import OUTPUTS as OUTS
from lepy.outputs.writer import OutputWriter, Plotter
import munch

# --- napari-specific glue ---------------------------------------------------
from napari_lepy._layer_io import (
    POI_NAMES, POI_COLORS,
    pois_to_array, array_to_keypoints,
    labels_to_mask, contour_to_polygon, recompute_contour,
)


logger = logging.getLogger(__name__)


# Default segmentation config, mirroring config.yml.
DEFAULT_SEG_CONFIG = {
    "method": "grabcut+otsu",
    "channel": "saturation",
    "rescale": 0.5,
    "ksize": None,
    "fill_holes": True,
}


class LEPYWidget(QWidget):
    """Dock widget that drives the original LEPY pipeline interactively."""

    # Layer name constants -- so we always read/write the same layers.
    L_RGB = "RGB"
    L_UV = "UV"
    L_GBUV = "GB-UV"
    L_CHANNEL = "Channel"
    L_MASK = "Mask"
    L_CONTOUR = "Contour"
    L_POI = "POI"
    L_DIST = "Distances"

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # The one and only LEPY Image instance.
        self._image: T.Optional[LepyImage] = None
        # Current px/mm scale from calibration or manual entry.
        self._scale: T.Optional[float] = None
        # Latest detected / edited PointsOfInterest.
        self._pois: T.Optional[PointsOfInterest] = None
        # Source file paths (used by exporter to name files).
        self._rgb_path: T.Optional[Path] = None
        self._uv_path: T.Optional[Path] = None

        self._build_ui()

    # =======================================================================
    # UI
    # =======================================================================
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

        # --- ① Load ---------------------------------------------------------
        gb = QGroupBox("① Load images")
        v = QVBoxLayout(gb)
        row = QHBoxLayout()
        self.btn_rgb = QPushButton("Load RGB…")
        self.btn_rgb.clicked.connect(self._on_load_rgb)
        row.addWidget(self.btn_rgb)
        self.btn_uv = QPushButton("Load UV…")
        self.btn_uv.clicked.connect(self._on_load_uv)
        self.btn_uv.setEnabled(False)
        row.addWidget(self.btn_uv)
        v.addLayout(row)
        self.lbl_loaded = QLabel("No image loaded.")
        self.lbl_loaded.setWordWrap(True)
        v.addWidget(self.lbl_loaded)
        layout.addWidget(gb)

        # --- ② Segment ------------------------------------------------------
        gb = QGroupBox("② Segmentation (lepy.segmentation.segment)")
        v = QVBoxLayout(gb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems([
            "otsu", "gauss_local",
            "grabcut", "grabcut+otsu",
            "unet", "flatbug",
        ])
        self.cb_method.setCurrentText("grabcut+otsu")
        row.addWidget(self.cb_method)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel:"))
        self.cb_channel = QComboBox()
        self.cb_channel.addItems(["saturation", "value", "hue", "uv"])
        row.addWidget(self.cb_channel)
        row.addWidget(QLabel("Rescale:"))
        self.sp_rescale = QDoubleSpinBox()
        self.sp_rescale.setRange(0.1, 1.0)
        self.sp_rescale.setSingleStep(0.1)
        self.sp_rescale.setValue(0.5)
        row.addWidget(self.sp_rescale)
        v.addLayout(row)
        row = QHBoxLayout()
        self.cb_fill = QCheckBox("Fill holes")
        self.cb_fill.setChecked(True)
        row.addWidget(self.cb_fill)
        row.addWidget(QLabel("ksize:"))
        self.sp_ksize = QSpinBox()
        self.sp_ksize.setRange(0, 99)
        self.sp_ksize.setSingleStep(2)
        self.sp_ksize.setSpecialValueText("off")
        row.addWidget(self.sp_ksize)
        v.addLayout(row)

        self.btn_segment = QPushButton("Run segmentation")
        self.btn_segment.clicked.connect(self._on_segment)
        self.btn_segment.setEnabled(False)
        v.addWidget(self.btn_segment)

        hint = QLabel(
            "✏️  Select the <b>Mask</b> layer in napari, use the brush "
            "(<kbd>P</kbd>) and eraser (<kbd>E</kbd>) to refine. Then:"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        v.addWidget(hint)

        self.btn_rerun_poi = QPushButton("Re-run POI from edited mask")
        self.btn_rerun_poi.clicked.connect(self._on_rerun_poi)
        self.btn_rerun_poi.setEnabled(False)
        v.addWidget(self.btn_rerun_poi)
        layout.addWidget(gb)

        # --- ③ Calibration --------------------------------------------------
        gb = QGroupBox("③ Calibration (lepy + scalebar)")
        v = QVBoxLayout(gb)
        row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Detect scalebar")
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_calibrate.setEnabled(False)
        row.addWidget(self.btn_calibrate)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Scale px/mm:"))
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.0, 999999.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setValue(0.0)
        self.sp_scale.setSpecialValueText("none")
        self.sp_scale.valueChanged.connect(self._on_manual_scale)
        row.addWidget(self.sp_scale)
        v.addLayout(row)
        layout.addWidget(gb)

        # --- ④ POI ----------------------------------------------------------
        gb = QGroupBox("④ POI & structural traits (lepy.poi)")
        v = QVBoxLayout(gb)
        hint = QLabel(
            "🖱️  Select the <b>POI</b> layer + select tool (<kbd>S</kbd>) "
            "to drag any point. Then:"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        v.addWidget(hint)

        self.btn_update_traits = QPushButton("Update traits from dragged POI")
        self.btn_update_traits.clicked.connect(self._on_update_traits_from_drag)
        self.btn_update_traits.setEnabled(False)
        v.addWidget(self.btn_update_traits)

        self.txt_traits = QTextEdit()
        self.txt_traits.setReadOnly(True)
        self.txt_traits.setMaximumHeight(170)
        self.txt_traits.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        v.addWidget(self.txt_traits)
        layout.addWidget(gb)

        # --- ⑤ Colour stats -------------------------------------------------
        gb = QGroupBox("⑤ Colour traits (Image.color_stats)")
        v = QVBoxLayout(gb)
        self.btn_color = QPushButton("Compute colour statistics")
        self.btn_color.clicked.connect(self._on_color_stats)
        self.btn_color.setEnabled(False)
        v.addWidget(self.btn_color)
        self.txt_color = QTextEdit()
        self.txt_color.setReadOnly(True)
        self.txt_color.setMaximumHeight(140)
        self.txt_color.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        v.addWidget(self.txt_color)
        layout.addWidget(gb)

        # --- ⑥ Export -------------------------------------------------------
        gb = QGroupBox("⑥ Export (lepy.OutputWriter + Plotter)")
        v = QVBoxLayout(gb)
        self.cb_plotter = QCheckBox("Also save the LEPY visual summary (PNG)")
        self.cb_plotter.setChecked(True)
        v.addWidget(self.cb_plotter)
        self.btn_export = QPushButton("Export all…")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        v.addWidget(self.btn_export)
        layout.addWidget(gb)

        layout.addStretch(1)

    # =======================================================================
    # Step ①: Load
    # =======================================================================
    def _on_load_rgb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open RGB image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All (*)")
        if not path:
            return
        path = Path(path)
        rgb = imread(str(path))
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = rgb.astype(np.uint8)

        # Build a LEPY Image instance bypassing the on-disk read() so we can
        # work with napari-loaded data.
        self._image = LepyImage(
            root=path.parent, key=path.stem, rgb_fname=path.name)
        self._image.rgb_im = rgb
        self._image.gray_im = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        self._rgb_path = path
        self._uv_path = None

        self._add_image(self.L_RGB, rgb)
        self._remove_layers([self.L_UV, self.L_GBUV, self.L_CHANNEL,
                             self.L_MASK, self.L_CONTOUR,
                             self.L_POI, self.L_DIST])

        self.lbl_loaded.setText(
            f"RGB: <b>{path.name}</b> ({rgb.shape[1]}×{rgb.shape[0]})")
        self.btn_uv.setEnabled(True)
        self.btn_segment.setEnabled(True)
        self.btn_calibrate.setEnabled(True)
        self.viewer.reset_view()

    def _on_load_uv(self) -> None:
        if self._image is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UV image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All (*)")
        if not path:
            return
        path = Path(path)
        uv_full = imread(str(path))
        # LEPY convention: take uv_channel_index from config (default 0).
        if uv_full.ndim == 3:
            uv = uv_full[..., 0]
        else:
            uv = uv_full
        uv = uv.astype(np.uint8)

        self._image.uv_fname = path.name
        self._image.uv_im = uv
        self._uv_path = path

        self._add_image(self.L_UV, uv, colormap="magenta", blending="additive")

        # GB-UV false-colour preview (Image.gbuv_im).
        try:
            gbuv = self._image.gbuv_im
            self._add_image(self.L_GBUV, gbuv)
        except AssertionError:
            pass

        self.lbl_loaded.setText(
            self.lbl_loaded.text() + f" &nbsp; | &nbsp; UV: <b>{path.name}</b>")

    # =======================================================================
    # Step ②: Segmentation
    # =======================================================================
    def _on_segment(self) -> None:
        if self._image is None:
            return
        cfg = munch.munchify({
            "method": self.cb_method.currentText(),
            "channel": self.cb_channel.currentText(),
            "rescale": float(self.sp_rescale.value()),
            "ksize": self.sp_ksize.value() if self.sp_ksize.value() > 0 else None,
            "fill_holes": self.cb_fill.isChecked(),
        })
        # Make sure ksize is odd if non-null (cv2.GaussianBlur requirement).
        if cfg["ksize"] is not None and cfg["ksize"] % 2 == 0:
            cfg["ksize"] += 1

        try:
            self._image.segment(config=cfg)
        except Exception as e:
            logger.exception("Segmentation failed")
            self._error("Segmentation failed", str(e))
            return

        self._refresh_segmentation_layers()

        # Auto-run POI on the fresh mask.
        self._run_pois_from_detection()
        self.btn_rerun_poi.setEnabled(True)
        self.btn_color.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_update_traits.setEnabled(True)

    def _refresh_segmentation_layers(self) -> None:
        """Push Image.chan / Image.mask / Image.contour to the viewer."""
        im = self._image
        if im.chan is not None:
            self._add_image(self.L_CHANNEL, im.chan, colormap="gray",
                            visible=False)
        if im.mask is not None:
            self._add_labels(self.L_MASK, im.mask.astype(np.uint8))
        if im.contour is not None and len(im.contour) > 0:
            poly = contour_to_polygon(im.contour)
            self._add_shapes(self.L_CONTOUR, [poly], shape_type="polygon",
                             edge_color="#00ff88", edge_width=2,
                             face_color="transparent")

    # =======================================================================
    # Mask brush workflow -> re-run POI
    # =======================================================================
    def _on_rerun_poi(self) -> None:
        """User edited the Mask Labels layer with the brush; pull the edited
        data back into ``Image.mask`` / ``Image.contour`` and re-run
        ``PointsOfInterest.detect``.
        """
        if self._image is None or self.L_MASK not in self.viewer.layers:
            return
        edited = labels_to_mask(self.viewer.layers[self.L_MASK].data)
        self._image.mask = edited
        self._image.contour = recompute_contour(edited)

        # Update the on-screen contour to match the new mask.
        if len(self._image.contour) > 0:
            self._add_shapes(self.L_CONTOUR,
                             [contour_to_polygon(self._image.contour)],
                             shape_type="polygon",
                             edge_color="#00ff88", edge_width=2,
                             face_color="transparent")

        # Clear any stale manual keypoints so .detect() runs again.
        self._image.key_points = None
        self._run_pois_from_detection()

    def _run_pois_from_detection(self) -> None:
        """Call ``Image.pois`` with detection (no key_points)."""
        if self._image is None or self._image.mask is None:
            return
        try:
            cfg = munch.munchify({"enabled": True})
            self._pois = self._image.pois(cfg, scale=self._scale)
        except Exception as e:
            logger.exception("POI detection failed")
            self._error("POI detection failed",
                        f"{e}\n\nTry editing the Mask with the brush first.")
            self._pois = None
            return
        self._show_pois()
        self._show_traits()

    # =======================================================================
    # POI drag -> update traits via from_keypoints
    # =======================================================================
    def _on_update_traits_from_drag(self) -> None:
        """User dragged points in the POI layer. Use those positions as
        ``Image.key_points`` so ``Image.pois`` builds a
        ``PointsOfInterest`` via ``from_keypoints`` rather than ``.detect``.
        """
        if self._image is None or self.L_POI not in self.viewer.layers:
            return
        arr = np.asarray(self.viewer.layers[self.L_POI].data)
        try:
            kp = array_to_keypoints(arr)
        except ValueError as e:
            self._error("Cannot read POI", str(e))
            return
        self._image.key_points = kp
        try:
            cfg = munch.munchify({"enabled": True})
            self._pois = self._image.pois(cfg, scale=self._scale)
        except Exception as e:
            logger.exception("POI from keypoints failed")
            self._error("POI update failed", str(e))
            return
        # Don't overwrite the dragged Points -- but refresh distance lines and
        # trait readout.
        self._draw_distance_lines()
        self._show_traits()

    # =======================================================================
    # Calibration
    # =======================================================================
    def _on_calibrate(self) -> None:
        if self._image is None or self._rgb_path is None:
            return
        # Find a scalebar template: bundled with the plugin, or fall back to
        # the working directory.
        candidates = [
            Path(__file__).parent / "scalebar_template.jpg",
            Path.cwd() / "scalebar_template.jpg",
        ]
        template_path = next((p for p in candidates if p.exists()), None)
        if template_path is None:
            self._error(
                "No scalebar template",
                "Place ``scalebar_template.jpg`` next to the plugin or in the "
                "working directory, or enter px/mm manually.")
            return
        cfg = munch.munchify({
            "enabled": True,
            "max_corners": 50,
            "roi_fraction": 0.15,
            "square_size": 1.0,
            "position": None,
            "template_path": str(template_path),
            "template_scale": 290.0,
        })
        try:
            res = self._image.calibrate(cfg)
        except Exception as e:
            logger.exception("Calibration failed")
            self._error("Calibration failed", str(e))
            return
        if res is None or res.scale is None:
            self._error("Calibration failed", "scalebar.Result was empty.")
            return
        self._scale = float(res.scale)
        self.sp_scale.blockSignals(True)
        self.sp_scale.setValue(self._scale)
        self.sp_scale.blockSignals(False)
        # Refresh trait readout in mm.
        self._show_traits()

    def _on_manual_scale(self, value: float) -> None:
        self._scale = float(value) if value > 0 else None
        if self._image is not None and self._image.mask is not None:
            # Recompute stats with the new scale (this updates calibrated
            # area / length entries in image.stats too).
            self._show_traits()

    # =======================================================================
    # Colour stats
    # =======================================================================
    def _on_color_stats(self) -> None:
        if self._image is None or self._image.mask is None:
            return
        try:
            stats = self._image.color_stats(binsize=2)
        except Exception as e:
            logger.exception("color_stats failed")
            self._error("Colour stats failed", str(e))
            return

        # Channel order matches Image.color_stats: R, G, B, UV (if has_uv),
        # RGB-UV mixed. Channels that were all zero are skipped by LEPY.
        all_names = ["Red", "Green", "Blue", "UV", "RGB-UV"]
        names = all_names[:len(stats.medians)]

        lines: list = []
        for name, q25, q75, median, iqr, shannon in zip(
                names, stats.q25s, stats.q75s, stats.medians,
                stats.iqrs, stats.shannon_indices):
            lines.append(
                f"{name:<7s}  median={median:6.1f}  Q25={q25:6.1f}  "
                f"Q75={q75:6.1f}  IQR={iqr:6.1f}  H={shannon:.2f}"
            )
        self.txt_color.setPlainText("\n".join(lines))

    # =======================================================================
    # Visualisation helpers
    # =======================================================================
    def _show_pois(self) -> None:
        if self._pois is None:
            return
        arr = pois_to_array(self._pois)
        colors = [POI_COLORS[n] for n in POI_NAMES]
        text = {"string": POI_NAMES, "color": "white",
                "size": 9, "translation": [-14, 0]}
        if self.L_POI in self.viewer.layers:
            layer = self.viewer.layers[self.L_POI]
            layer.data = arr
            layer.face_color = colors
            try:
                layer.text = text
            except Exception:
                pass
        else:
            self.viewer.add_points(
                arr, name=self.L_POI, size=16,
                face_color=colors,
                border_color="white", border_width=0.15,
                text=text,
            )
        self._draw_distance_lines()

    def _draw_distance_lines(self) -> None:
        """Reflect the four named distances as line shapes (also drawn in
        Plotter._plot_struc_traits).
        """
        pois = self._pois
        if pois is None:
            return
        pairs = [
            (pois.inner_top_l, pois.inner_top_r),  # body width
            (pois.body_top,    pois.body_bot),     # body length
            (pois.inner_top_l, pois.outer_l),      # left FWL
            (pois.inner_top_r, pois.outer_r),      # right FWL
        ]
        lines = [np.array([[a.row, a.col], [b.row, b.col]],
                          dtype=float) for a, b in pairs]
        self._add_shapes(self.L_DIST, lines, shape_type="line",
                         edge_color="#00d4ff", edge_width=2)

    def _show_traits(self) -> None:
        """Render the structural-trait table from ``image.stats``."""
        if self._image is None:
            return
        s = self._image.stats
        unit = "mm" if self._scale else "px"
        unit_a = "mm²" if self._scale else "px²"
        rows: list = []
        for key, label in [
            (OUTS.poi.dist.inner_outer_l, "Left FWL"),
            (OUTS.poi.dist.inner_outer_r, "Right FWL"),
            (OUTS.poi.dist.inner,         "Body width"),
            (OUTS.poi.dist.body,          "Body length"),
            (OUTS.contour.width_calibrated if self._scale else None,
                                          "Wing span"),
            (OUTS.poi.area.wing_l,        "Left wing area"),
            (OUTS.poi.area.wing_r,        "Right wing area"),
            (OUTS.poi.area.body,          "Body area"),
        ]:
            if key is None or key not in s:
                continue
            val = s[key]
            u = unit_a if "area" in key else unit
            rows.append(f"  {label:<18s} {val:>10.2f} {u}")
        if self._scale:
            rows.insert(0, f"  Scale              {self._scale:>10.2f} px/mm")
        self.txt_traits.setPlainText("\n".join(rows) if rows else "(no traits yet)")

    # =======================================================================
    # Export -> uses lepy.OutputWriter and lepy.Plotter unchanged
    # =======================================================================
    def _on_export(self) -> None:
        if self._image is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not folder:
            return
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)

        # Make sure mask/contour in the Image reflect the *current* napari
        # layer state (the user might have brushed without re-running POI).
        if self.L_MASK in self.viewer.layers:
            self._image.mask = labels_to_mask(
                self.viewer.layers[self.L_MASK].data)
            self._image.contour = recompute_contour(self._image.mask)

        # Make sure stats are up to date with any dragged POI.
        if (self.L_POI in self.viewer.layers
                and np.asarray(self.viewer.layers[self.L_POI].data).shape[0]
                == len(POI_NAMES)):
            self._image.key_points = array_to_keypoints(
                np.asarray(self.viewer.layers[self.L_POI].data))
            try:
                self._pois = self._image.pois(
                    munch.munchify({"enabled": True}), scale=self._scale)
            except Exception as e:
                logger.warning("Could not refresh POI from layer: %s", e)

        # 1) Save raw layers as images.
        if self._image.rgb_im is not None:
            imsave(str(out / "rgb.tif"), self._image.rgb_im,
                   check_contrast=False)
        if self._image.uv_im is not None:
            imsave(str(out / "uv.tif"), self._image.uv_im,
                   check_contrast=False)
        if self._image.chan is not None:
            imsave(str(out / "channel.tif"), self._image.chan,
                   check_contrast=False)
        if self._image.mask is not None:
            imsave(str(out / "mask.tif"),
                   (self._image.mask * 255).astype(np.uint8),
                   check_contrast=False)
        try:
            gbuv = self._image.gbuv_im
            imsave(str(out / "gbuv.tif"), gbuv, check_contrast=False)
        except AssertionError:
            pass

        # 2) Use the *original* LEPY OutputWriter for the trait JSON/CSV.
        writer = OutputWriter(str(out), store_to_csv=True, config=None)
        rgb_path_for_writer = self._rgb_path or (
            self._image.root / (self._image.rgb_fname or "image.jpg"))
        try:
            writer(str(rgb_path_for_writer), self._image.stats)
        except Exception as e:
            logger.warning("OutputWriter failed: %s", e)

        # 3) Use the *original* LEPY Plotter for the visual summary
        #    (the big multi-panel figure in the LEPY pipeline diagram).
        if self.cb_plotter.isChecked():
            try:
                # Plotter wants a calib_result; if we never ran calibration
                # we fake a minimal one so the plot still renders.
                calib_result = self._fake_calib_result()
                plotter = Plotter(str(out), plot_interm=False,
                                  save_contours=True)
                plotter.plot(self._image,
                             pois=self._pois,
                             calib_result=calib_result)
            except Exception as e:
                logger.warning("Plotter failed: %s", e)

        # 4) POI table as a stand-alone CSV (LEPY column names).
        if self._pois is not None:
            kp = array_to_keypoints(pois_to_array(self._pois))
            with open(out / "poi.json", "w") as f:
                json.dump(kp, f, indent=2)

        QMessageBox.information(
            self, "Export complete",
            f"Outputs written to:\n{out}\n\n"
            "Files: rgb.tif, uv.tif (if loaded), channel.tif, mask.tif, "
            "gbuv.tif (if UV), contours/, json/, stats.csv, poi.json, "
            "visualisations/ (visual summary)."
        )

    def _fake_calib_result(self):
        """Build a minimal stand-in for ``scalebar.Result`` so Plotter.plot
        works even without calibration."""
        class _Pos:
            x = y = width = height = 0

            def crop(self, im):
                return im

        class _Dist:
            corners = np.zeros((0, 2))

        class _Stub:
            scale = self._scale or 0.0
            position = _Pos()
            distances = _Dist()
            scalebar = np.zeros((10, 10), dtype=np.uint8)
            match = None
            images = None
            roi_fraction = 0.0
            scalebar_location = None

            def best_corners(self):
                return np.array([], dtype=int), []
        return _Stub()

    # =======================================================================
    # Tiny layer wrappers (add-or-update pattern)
    # =======================================================================
    def _add_image(self, name: str, data: np.ndarray, **kw) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_image(data, name=name, **kw)

    def _add_labels(self, name: str, data: np.ndarray) -> None:
        data = data.astype(np.int32)
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _add_shapes(self, name: str, data, **kw) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_shapes(data, name=name, **kw)

    def _remove_layers(self, names: T.Iterable[str]) -> None:
        for n in names:
            if n in self.viewer.layers:
                self.viewer.layers.remove(n)

    def _error(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)
