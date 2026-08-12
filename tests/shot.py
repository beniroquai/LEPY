"""Render screenshots of the review widget (needs a visible Qt/GL session).

    python tests/shot.py [DATA_DIR] [SHOT_DIR]

Opens a real napari window briefly, writes three PNGs, then closes it.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import napari
from qtpy.QtWidgets import QApplication

from napari_lepy.review._review_widget import LEPYReviewWidget

ROOT = Path(sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("LEPY_TEST_DATA",
                                Path.home() / "Downloads/Lepy_data")).expanduser()
SHOTS = Path(sys.argv[2] if len(sys.argv) > 2
             else Path(tempfile.gettempdir()) / "lepy_shots")
SHOTS.mkdir(parents=True, exist_ok=True)

viewer = napari.Viewer(show=True)
viewer.window._qt_window.resize(1500, 950)
widget = LEPYReviewWidget(viewer)
viewer.window.add_dock_widget(widget, area="right", name="LEPY Review")
widget.open_run(ROOT / "Napari_Lepy_Data",
                Path(tempfile.gettempdir()) / "lepy_shot_out")


def settle(n: int = 40) -> None:
    for _ in range(n):
        QApplication.processEvents()


settle()
viewer.screenshot(str(SHOTS / "01_full.png"), canvas_only=False, flash=False)

poi = np.asarray(viewer.layers[widget.L_POI].data)
viewer.camera.center = (0, float(poi[:, 0].mean()), float(poi[:, 1].mean()))
viewer.camera.zoom = 0.9
settle()
viewer.screenshot(str(SHOTS / "02_poi_zoom.png"), canvas_only=True, flash=False)

widget.cb_triage.setChecked(True)
viewer.reset_view()
settle()
viewer.screenshot(str(SHOTS / "03_qc_triage.png"), canvas_only=True, flash=False)

print(f"screenshots written to {SHOTS}")
viewer.close()
