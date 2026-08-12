"""Drive the LEPY review widget in a real (hidden) napari viewer.

    python tests/test_review_gui.py [DATA_DIR] [OUT_DIR]

Needs a working Qt/OpenGL session (it builds a real napari viewer, hidden).
Modal dialogs are stubbed out so nothing blocks. Screenshots are skipped
because a hidden canvas has no GL context -- run tests/shot.py for those.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import napari
from napari_lepy.review._review_widget import LEPYReviewWidget
from napari_lepy.review import status as st

ROOT = Path(sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("LEPY_TEST_DATA",
                                Path.home() / "Downloads/Lepy_data")).expanduser()
DATA = ROOT / "Napari_Lepy_Data"
OUT = Path(sys.argv[2] if len(sys.argv) > 2
           else Path(tempfile.gettempdir()) / "lepy_gui_out")
if not DATA.exists():
    sys.exit(f"no test data at {DATA} -- pass a path or set $LEPY_TEST_DATA")
if OUT.exists():
    shutil.rmtree(OUT)

ok = True
def check(label, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    tail = str(extra)
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f" :: {tail}" if tail else ""))

# Modal dialogs would block a headless run -- record them instead.
import napari_lepy.review._review_widget as rw
DIALOGS = []
rw.QMessageBox.warning = staticmethod(
    lambda *a, **k: DIALOGS.append(("warning", a[1] if len(a) > 1 else "")))
rw.QMessageBox.information = staticmethod(
    lambda *a, **k: DIALOGS.append(("info", a[1] if len(a) > 1 else "")))
rw.QMessageBox.critical = staticmethod(
    lambda *a, **k: DIALOGS.append(("critical", a[1] if len(a) > 1 else "")))

viewer = napari.Viewer(show=False)
w = LEPYReviewWidget(viewer)
viewer.window.add_dock_widget(w, area="right", name="LEPY Review")

# ------------------------------------------------- keyboard shortcut safety
# napari resolves the ACTIVE LAYER's keymap before the viewer's, so any review
# shortcut that a layer also binds is dead whenever that layer is selected.
probe = napari.Viewer(show=False)
layer_keys = set()
for probe_layer in (probe.add_points(np.zeros((1, 2))),
                    probe.add_shapes([np.array([[0, 0], [1, 1], [0, 1]])],
                                     shape_type="polygon"),
                    probe.add_labels(np.zeros((4, 4), dtype=int)),
                    probe.add_image(np.zeros((4, 4)))):
    layer_keys |= {str(k).lower() for k in (type(probe_layer).class_keymap or {})}
    layer_keys |= {str(k).lower() for k in (probe_layer.keymap or {})}
probe.close()
review_keys = {rw.KEY_CONFIRM, rw.KEY_FLAG, rw.KEY_SAVE,
               rw.KEY_QC, rw.KEY_RESET, rw.KEY_UNREVIEW}
clashes = sorted(k for k in review_keys if k.lower() in layer_keys)
check("no shortcut collides with a napari layer keymap", not clashes, clashes)
check("all six shortcuts are distinct", len(review_keys) == 6, review_keys)
check("shortcuts registered on the viewer",
      review_keys | {"Right", "Left"} <= {str(k) for k in viewer.keymap},
      {str(k) for k in viewer.keymap})

# ------------------------------------------------------------------ open
w.open_run(DATA, OUT)
check("run opened", w.run is not None and len(w.run.codes) == 13)
check("db seeded with every code", len(w.db.entries) == 13, len(w.db.entries))
check("failed code pre-marked", w.db.status("EcEs-Lep-00462") == st.FAILED)
check("review_status.csv created", (OUT / "review_status.csv").exists())

# batch of 5 -> 3 batches
w.sp_batch_size.setValue(5)
check("batch split", len(w.run.batches(5)) == 3)
check("view holds first batch", len(w._view_codes) == 5, w._view_codes)
check("landed on first specimen", w._record.code == w._view_codes[0], w._record.code)

# ----------------------------------------------------------------- layers
check("Specimen layer", w.L_IMAGE in viewer.layers, list(viewer.layers))
check("QC layer", w.L_QC in viewer.layers)
check("Contour layer", w.L_CONTOUR in viewer.layers)
check("POI layer", w.L_POI in viewer.layers)
poi_layer = viewer.layers[w.L_POI]
check("8 POI points", len(poi_layer.data) == 8, len(poi_layer.data))
check("8 distinct face colours", len({tuple(c) for c in poi_layer.face_color}) == 8)
check("per-point symbols applied",
      len({str(s) for s in np.atleast_1d(poi_layer.symbol)}) == 5,
      {str(s) for s in np.atleast_1d(poi_layer.symbol)})
qc = viewer.layers[w.L_QC]
check("QC fitted inside the specimen extent",
      qc.extent.world[1][0] <= viewer.layers[w.L_IMAGE].extent.world[1][0] + 1
      and qc.extent.world[1][1] <= viewer.layers[w.L_IMAGE].extent.world[1][1] + 1,
      (qc.extent.world[1], viewer.layers[w.L_IMAGE].extent.world[1]))


# ------------------------------------------------------------------ triage
w.cb_triage.setChecked(True)
check("triage hides the specimen", not viewer.layers[w.L_IMAGE].visible)
check("triage shows the QC figure", viewer.layers[w.L_QC].visible)
w.cb_triage.setChecked(False)
check("leaving triage restores the overlay", viewer.layers[w.L_IMAGE].visible)

# ------------------------------------------------------------ confirm + step
first = w._record.code
w._mark(st.OK, advance=True)
check("marked OK", w.db.status(first) == st.OK, w.db.status(first))
check("advanced", w._record.code != first, w._record.code)
from napari_lepy.review.status import ReviewStatusDB
check("verdict persisted to disk",
      ReviewStatusDB(OUT / "review_status.csv").status(first) == st.OK)

second = w._record.code
w._step(1)
check("arrow step forward", w._record.code == w._view_codes[2], w._record.code)
w._step(-1)
check("arrow step back", w._record.code == second, w._record.code)

# ------------------------------------------------------------- correct POI
target = "EcEs-Lep-00034"
w._position = w._view_codes.index(target)
w._load_current()
check("jumped to target", w._record.code == target)

before = np.asarray(viewer.layers[w.L_POI].data).copy()
moved = before.copy()
moved[2] += [500, -700]                    # outer_l
viewer.layers[w.L_POI].data = moved
check("edit marked dirty", w._poi_dirty)
check("status chip shows unsaved", "unsaved" in w.lbl_status.text(), w.lbl_status.text())

w._on_save()
check("saved as corrected", w.db.status(target) == st.CORRECTED)
entry = w.db.get(target)
check("poi_corrected flag", entry.poi_corrected is True)
check("mask untouched", entry.mask_corrected is False)
check("no re-analysis needed for a POI-only fix", entry.needs_reanalysis is False)
check("one POI counted", entry.n_poi_moved == 1, entry.n_poi_moved)
check("shift recorded", abs(entry.max_poi_shift_px - np.hypot(500, 700)) < 1,
      entry.max_poi_shift_px)
check("json written", (OUT / "stats_json" / f"{target}.json").exists())
check("contour written", (OUT / "contours_txt" / f"{target}.txt").exists())
check("no mask png for a POI-only fix",
      not (OUT / "masks_png" / f"{target}.png").exists())
check("corrected_codes.txt lists it",
      (OUT / "corrected_codes.txt").read_text().split() == [target],
      (OUT / "corrected_codes.txt").read_text().split())
check("reanalysis_codes.txt empty",
      (OUT / "reanalysis_codes.txt").read_text().strip() == "")
check("traits panel shows LEPY and exact columns",
      "exact" in w.txt_traits.toPlainText(), w.txt_traits.toPlainText()[:80])

# reload the same specimen -> corrected POI must come back
w._step(1); w._step(-1)
check("corrected POI reloaded",
      np.allclose(np.asarray(viewer.layers[w.L_POI].data)[2], moved[2]),
      np.asarray(viewer.layers[w.L_POI].data)[2])

# ------------------------------------------------------------ correct mask
w._on_simplify()
check("simplify marks the contour dirty", w._contour_dirty)
n_simple = len(viewer.layers[w.L_CONTOUR].data[0])
check("outline simplified", n_simple < 500, n_simple)
w._on_save()
entry = w.db.get(target)
check("mask_corrected flag set", entry.mask_corrected is True)
check("re-analysis now required", entry.needs_reanalysis is True)
check("poi_corrected flag survived the second save", entry.poi_corrected is True)
check("mask png written", (OUT / "masks_png" / f"{target}.png").exists())
check("reanalysis_codes.txt lists it",
      (OUT / "reanalysis_codes.txt").read_text().split() == [target])

# a further POI-only save must not clear needs_reanalysis
d = np.asarray(viewer.layers[w.L_POI].data).copy(); d[3] += [5, 5]
viewer.layers[w.L_POI].data = d
w._on_save()
check("needs_reanalysis is sticky", w.db.get(target).needs_reanalysis is True)

# ---------------------------------------------------------- brush workflow
w._on_to_brush()
check("brush layer created", w.L_MASK in viewer.layers)
mask = np.asarray(viewer.layers[w.L_MASK].data)
check("brush mask non-empty", mask.sum() > 0)
viewer.layers[w.L_MASK].data = np.zeros_like(mask)
DIALOGS.clear()
w._on_from_brush()   # empty -> must refuse, not crash
check("empty brush mask refused", len(viewer.layers[w.L_CONTOUR].data[0]) == n_simple)
check("empty brush mask warns the user", DIALOGS and DIALOGS[0][0] == "warning", DIALOGS)

# deleting a POI must be refused, not silently written
d = np.asarray(viewer.layers[w.L_POI].data)
viewer.layers[w.L_POI].data = d[:7]
DIALOGS.clear()
w._on_save()
check("7-point POI layer refused", DIALOGS and "POI" in str(DIALOGS[0][1]), DIALOGS)
w._on_reset_to_lepy()
check("reset restores 8 POIs", len(viewer.layers[w.L_POI].data) == 8)
check("reset clears the dirty flags", not w._poi_dirty and not w._contour_dirty)

# ------------------------------------------------------------------ filter
w.cb_filter.setCurrentIndex(1)          # "Not reviewed"
check("filter applied", all(w.db.status(c) == st.UNREVIEWED for c in w._view_codes),
      [(c, w.db.status(c)) for c in w._view_codes])
n_open = len(w._view_codes)
w._mark(st.OK, advance=True)
check("filtered list shrinks after a verdict", len(w._view_codes) == n_open - 1,
      len(w._view_codes))
w.cb_filter.setCurrentIndex(0)

# ----------------------------------------------------------- failed record
w.sp_batch_size.setValue(100)
w._position = w._view_codes.index("EcEs-Lep-00462")
w._load_current()
check("failed specimen loads its image", w.L_IMAGE in viewer.layers)
check("no POI layer for a failed specimen", w.L_POI not in viewer.layers)
check("notice explains the failure", "no result" in w.lbl_notice.text().lower(),
      w.lbl_notice.text())
check("shape taken from the image when stats are missing",
      w._shape == (900, 1600), w._shape)
w._on_paint_mask()
check("paint layer created for rescue", w.L_MASK in viewer.layers)
m = np.asarray(viewer.layers[w.L_MASK].data)
check("paint layer matches the image shape", m.shape == w._shape, m.shape)

# paint a real moth silhouette: the 00025 contour, scaled into this image
from napari_lepy.review import recompute as rc2
donor = w.run.record("EcEs-Lep-00025").contour_px().astype(float)
donor[:, 0] *= w._shape[1] / 6000.0
donor[:, 1] *= w._shape[0] / 4000.0
m = rc2.rasterize_contour(donor.astype(np.int32), w._shape)
viewer.layers[w.L_MASK].data = m
DIALOGS.clear()
w._on_from_brush()
check("no dialog when the painted mask is valid", not DIALOGS, DIALOGS)
check("outline derived from the painted mask",
      len(w._current_contour()) >= 4, len(w._current_contour()))
w._on_redetect_poi()
check("POIs detected on the rescued outline",
      w.L_POI in viewer.layers and len(viewer.layers[w.L_POI].data) == 8)
w._on_save()
check("rescued specimen saved", w.db.status("EcEs-Lep-00462") == st.CORRECTED)
check("rescued stats have POIs",
      w.corrected_stats["EcEs-Lep-00462"].get("poi_body_top_x") is not None)

# ------------------------------------------------------------------ export
w._on_export = w._on_export  # noqa
written = w.writer.write_tables(w.corrected_stats)
check("export wrote tables", len(written) >= 5, [p.name for p in written])
import csv
with open(OUT / "stats_combined_merged.csv", newline="") as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))
check("merged table now includes the rescued specimen",
      "EcEs-Lep-00462" in {r["Code"] for r in rows}, len(rows))

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
