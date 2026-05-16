# napari-lepy

**Interactive [napari][napari] plugin for [LEPY][lepy] — the Lepidoptera phenotyping pipeline.**

`napari-lepy` is a thin wrapper around the original LEPY package. The full
LEPY API (`lepy.Image`, `lepy.Worker`, `lepy.PointsOfInterest`,
`lepy.OutputWriter`, `lepy.Plotter`, …) is shipped **unmodified** and remains
fully usable as a stand-alone command-line tool. The plugin adds a graphical
front-end that lets you:

- run every LEPY step **inside napari** with its results shown as layers,
- **fix the binary mask by hand** with napari's brush / eraser whenever the
  U-Net / Otsu / GrabCut output isn't perfect,
- **re-run point-of-interest detection** on the corrected mask,
- **drag any POI** to a new location and have the structural traits
  recomputed automatically (uses LEPY's existing
  `PointsOfInterest.from_keypoints` hook — no monkey-patching),
- **export everything**: every layer as TIFF, plus the original LEPY
  CSV / JSON outputs and the standard LEPY visual-summary PNG.

[napari]: https://napari.org
[lepy]: https://github.com/Jena-CompPhylo-Lab/LEPY

---

## Pipeline overview

| Step | Where it runs | napari layer(s) |
|------|---------------|-----------------|
| ① Load RGB (+ optional UV) | `Image.read` + manual file dialog | `RGB`, `UV`, `GB-UV` |
| ② Segment | `Image.segment` → `lepy.segmentation.segment` | `Channel`, `Mask` (editable), `Contour` |
| ③ Edit mask by hand | napari brush / eraser | `Mask` |
| ④ Re-run POI | `Image.pois` → `PointsOfInterest.detect` | `POI`, `Distances` |
| ⑤ Drag a POI | napari select tool | `POI` |
| ⑥ Update traits | `Image.pois` → `PointsOfInterest.from_keypoints` | trait readout |
| ⑦ Calibrate | `Image.calibrate` (scalebar package) | scale field |
| ⑧ Colour stats | `Image.color_stats` | colour readout |
| ⑨ Export | `lepy.OutputWriter` + `lepy.Plotter` + layer TIFFs | files on disk |

The naming of every layer, statistic, CSV column, and JSON key matches the
upstream LEPY conventions (`poi_body_top_x`, `poi_dist_inner_outer_l`,
`contour_width_calibrated`, etc.) so any downstream analysis that worked with
LEPY's batch output works with napari-lepy's output too.

---

## Installation

The plugin works on Python 3.9+.

### Option A: install everything from this repo

```bash
git clone <this-repo>
cd napari-lepy
pip install -e .
```

This installs:

- `lepy` (the original package, included verbatim),
- `napari_lepy` (the plugin),
- runtime dependencies: napari, numpy, scipy, scikit-image, opencv-python,
  matplotlib, scalebar.

### Option B: optional binarisation backends

LEPY ships several segmentation back-ends. The plugin works out of the box
with `otsu`, `gauss_local`, `grabcut`, and `grabcut+otsu`. The deep-learning
backends are optional and only loaded when you select them:

```bash
# U-Net background remover
pip install backgroundremover

# flat-bug (requires a manual install from git)
pip install git+https://github.com/darsa-group/flat-bug
```

### Verify

```bash
napari
# Plugins menu → "LEPY Pipeline"
```

You should see a dock widget on the right with sections ① through ⑥.

---

## Tutorial

This walkthrough uses the bundled example image (`examples/example.jpg`).

### 1 · Load an image

1. Click **Load RGB…** and pick `examples/example.jpg`.
2. (Optional) Click **Load UV…** and pick a paired UV image. A `GB-UV` false-
   colour layer is created automatically.

After loading you'll see one or three image layers in the napari layer list.

### 2 · Run the initial segmentation

In the **② Segmentation** group of the widget:

- choose a **Method** — `otsu` is fastest, `grabcut+otsu` is the LEPY default,
  `unet` requires `backgroundremover`, `flatbug` requires `flat-bug`;
- choose the **Channel** (`saturation` works well for most specimens);
- leave **Rescale** at 0.5 and **Fill holes** ticked for default LEPY behaviour;
- click **Run segmentation**.

Four new layers appear:

- `Channel` — the channel that was binarised (hidden by default);
- `Mask` — a **Labels** layer with the binary mask;
- `Contour` — the largest external contour as a polygon;
- `POI` — eight coloured points for the detected landmarks;
- `Distances` — four light-blue lines (left/right FWL, body length, body width).

### 3 · Fix the mask with the brush

If the mask is wrong somewhere (e.g. the U-Net missed a wing tip or included
shadow), correct it manually:

1. Click the **Mask** layer in the napari layer list to select it.
2. In the layer-controls panel on the left, click the **paint brush** icon
   (or press <kbd>P</kbd>).
3. To **add** to the foreground, make sure the **label** value is `1` and
   paint over the missing area.
4. To **erase** false foreground, set the label to `0` (or use the **eraser**
   tool with <kbd>E</kbd>).
5. Adjust the **brush size** in the layer controls as needed (1–200 px).

When done, click **Re-run POI from edited mask** in the widget. The plugin:

- pulls the edited Labels data back into `Image.mask`,
- recomputes the contour with `cv2.findContours`,
- clears any manual keypoints (`Image.key_points = None`),
- calls `Image.pois(...)` which under the hood runs
  `PointsOfInterest.detect` on the corrected mask,
- redraws the `POI`, `Contour`, and `Distances` layers,
- updates the structural-trait table.

Repeat as many times as needed.

### 4 · Drag a POI

If a point landed in the wrong place (a common failure of automatic POI
detection on damaged specimens):

1. Click the **POI** layer in the layer list.
2. In the layer-controls panel, click the **select points** icon
   (arrow / <kbd>S</kbd>).
3. Click and drag any landmark to its correct location.
4. Click **Update traits from dragged POI** in the widget.

Under the hood this writes the new positions into `Image.key_points` (the
LEPY-native dict with keys like `poi_body_top_x`), then calls `Image.pois(...)`
which now routes through `PointsOfInterest.from_keypoints` instead of
`.detect`. All distances, areas and downstream stats are recomputed with the
manual positions — no LEPY code modified.

You can mix brush and drag freely: edit the mask → re-run POI → drag a few
points → update traits → export.

### 5 · Calibrate (optional)

There are two paths:

- **Automatic** — click **Detect scalebar** in the *③ Calibration* group. The
  plugin calls `Image.calibrate(...)` which uses the original `scalebar`
  package and the bundled `scalebar_template.jpg`. The detected px/mm value
  is written into the field below.
- **Manual** — type a known px/mm value directly into the **Scale px/mm**
  field. Set it to 0 to switch back to "pixels".

Any traits already in the readout are immediately recomputed with the new
scale.

### 6 · Colour statistics

Click **Compute colour statistics** in the *⑤ Colour traits* group. The
plugin runs `Image.color_stats(binsize=2)`. This populates `Image.stats` with
all the colour-related columns LEPY writes to its CSV (red/green/blue/UV
quartiles, Shannon and Simpson diversity, chromaticity, luminance, hue,
saturation, intensity). A compact summary is displayed in the widget.

### 7 · Export

Click **Export all…** and choose a folder. The plugin writes:

```
<folder>/
├── rgb.tif           # current RGB layer
├── uv.tif            # if UV was loaded
├── channel.tif       # binarisation channel
├── mask.tif          # FINAL mask (after any brush edits)
├── gbuv.tif          # GB-UV false-colour (if UV)
├── stats.csv         # ← original LEPY CSV (tab-separated, LEPY columns)
├── json/             # ← per-image JSON dump of Image.stats
│   └── <stem>.json
├── contours/         # ← LEPY-format YOLO-style contour text
│   └── <stem>.txt
├── visualisations/   # ← original LEPY visual-summary PNG
│   └── <stem>.png
└── poi.json          # ← all 8 POI in LEPY's key_points format
```

Both the CSV and the visual summary are produced by the **original LEPY**
`OutputWriter` and `Plotter` classes, so files are byte-compatible with
LEPY's batch outputs.

### 8 · Round-trip with command-line LEPY

The `poi.json` file uses LEPY's native `key_points` schema. To reprocess a
batch with your manual corrections:

```bash
python main.py \
    --folder my_images/ \
    --config config.yml \
    --key-points poi.json
```

LEPY will detect that `key_points` are available and call
`PointsOfInterest.from_keypoints` instead of `.detect` for those specimens.

---

## Layers reference

| Layer name | napari type | What it is | Editable? |
|------------|-------------|------------|-----------|
| `RGB`      | Image       | Original RGB image | – |
| `UV`       | Image       | UV channel (magenta colormap) | – |
| `GB-UV`    | Image       | False-colour Green=R, Blue=G, UV=B | – |
| `Channel`  | Image       | The channel passed to the binariser | – |
| `Mask`     | Labels      | Binary mask | **yes — brush / eraser** |
| `Contour`  | Shapes      | Largest contour polygon | – |
| `POI`      | Points      | 8 landmark points (colour-coded) | **yes — drag** |
| `Distances`| Shapes      | Line segments for the 4 named distances | – |

POI colour key:

- **cyan** — `body_top`, `body_bot`
- **magenta** — `outer_l`, `outer_r` (wingtips)
- **yellow** — `inner_top_l`, `inner_top_r` (shoulders)
- **orange** — `inner_bot_l`, `inner_bot_r` (inner wing bottom)

---

## API: scripting the plugin

The plugin re-exports the LEPY API at the package level:

```python
from napari_lepy import Image, PointsOfInterest, Worker, OutputWriter, Plotter
```

…so any code written for LEPY works without changes. In addition, the bridge
helpers are public:

```python
from napari_lepy._layer_io import (
    POI_NAMES,         # canonical order of POI in a napari Points layer
    pois_to_array,     # PointsOfInterest -> (8, 2) [row, col] array
    array_to_keypoints, # (8, 2) array  -> LEPY key_points dict
    recompute_contour, # binary mask -> largest contour (x, y)
    contour_to_polygon # (N, 2) (x, y) -> (N, 2) (row, col) for napari
)
```

Example — bypass the widget and drive everything from a script:

```python
import napari
from napari_lepy._widget import LEPYWidget

viewer = napari.Viewer()
widget = LEPYWidget(viewer)
viewer.window.add_dock_widget(widget, area="right")
napari.run()
```

---

## How the manual-correction loop works (internal)

The plugin never reimplements LEPY logic. The two manual workflows hook into
LEPY's existing extensibility points:

1. **Brush edit → re-run POI**
   - User paints / erases the `Mask` Labels layer.
   - Widget reads the edited array, writes it to `Image.mask`.
   - Widget calls `cv2.findContours` to refresh `Image.contour`.
   - Widget sets `Image.key_points = None`.
   - Widget calls `Image.pois(config, scale=scale)` — this is **the same
     method** the original `Worker` calls; it routes to
     `PointsOfInterest.detect(...)` because `key_points` is `None`.

2. **Drag a POI → update traits**
   - User drags points in the `POI` napari Points layer.
   - Widget converts the (8, 2) array of `[row, col]` positions into a dict
     with LEPY column-name keys (`poi_body_top_x`, …) via
     `array_to_keypoints`.
   - Widget writes that dict to `Image.key_points`.
   - Widget calls `Image.pois(...)` — which now routes to
     `PointsOfInterest.from_keypoints(...)` because `key_points` is set.

Both paths produce a `PointsOfInterest` instance whose `.stats`,
`.distances()`, and `.areas()` write the same LEPY-named entries into
`Image.stats` as the command-line pipeline would.

---

## Project layout

```
napari-lepy/
├── lepy/                     ← original LEPY package, unmodified*
│   ├── image.py
│   ├── poi.py
│   ├── segmentation.py
│   ├── binarization/
│   ├── outputs/
│   └── worker.py
├── napari_lepy/              ← the plugin
│   ├── __init__.py
│   ├── napari.yaml
│   ├── _widget.py            ← the dock widget
│   ├── _layer_io.py          ← bridge helpers (POI ↔ array, mask ↔ Labels)
│   └── scalebar_template.jpg ← bundled for one-click calibration
├── examples/
│   ├── example.jpg
│   └── uv_red.jpg
├── config.yml                ← LEPY default config (for the CLI)
├── scalebar_template.jpg
├── pyproject.toml
├── LICENSE
└── README.md
```

\* `lepy/binarization/__init__.py` was changed in one place: the `unet` and
`flatbug` imports are now lazy (moved inside the `binarize` function), so
the plugin remains usable without the optional `backgroundremover` / `flat-bug`
dependencies. All function signatures and behaviour are unchanged.

---

## Troubleshooting

- **"POI detection failed"** — usually means the mask is too noisy. Use the
  brush to clean it up (remove disconnected speckles, fill the body) and
  re-run POI.
- **Brush has no effect** — make sure the **Mask** layer is selected and the
  current label value is `1` (for adding) or `0` (for removing). Click the
  colour swatch in the layer controls to change.
- **Dragged POI snaps back after "Update traits"** — make sure exactly 8
  points are present in the POI layer. Use Ctrl+Z if you accidentally added
  or removed a point.
- **Calibration fails** — check that `scalebar_template.jpg` is bundled with
  the plugin (it is, by default). The `scalebar` package needs a high-
  resolution image with a visible black-and-white square ruler.
- **`unet` method fails to import** — install `backgroundremover` as
  described above.

---

## Licence

GPL-3.0-or-later (same as upstream LEPY).
