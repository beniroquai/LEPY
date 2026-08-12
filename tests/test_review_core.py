"""Headless check of the review core against a real LepyLoop run.

    python tests/test_review_core.py [DATA_DIR] [OUT_DIR]

DATA_DIR is the folder that was fed to LepyLoop (the ``*_Lepy_Execution_*``
sibling is discovered automatically); it defaults to ``$LEPY_TEST_DATA``.
No pytest needed -- it prints PASS/FAIL lines and exits non-zero on failure.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from napari_lepy.review.dataset import LepyRun, read_contour_txt, contour_to_txt
from napari_lepy.review.status import ReviewStatusDB, CORRECTED, OK, FAILED
from napari_lepy.review.export import CorrectionWriter, default_header
from napari_lepy.review import recompute as rc
from napari_lepy.review import palette as pal

DATA = Path(sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("LEPY_TEST_DATA",
                                Path.home() / "Downloads/Lepy_data")).expanduser()
OUT = Path(sys.argv[2] if len(sys.argv) > 2
           else Path(tempfile.gettempdir()) / "lepy_review_out")
if not DATA.exists():
    sys.exit(f"no test data at {DATA} -- pass a path or set $LEPY_TEST_DATA")
if OUT.exists():
    shutil.rmtree(OUT)

ok = True
def check(label, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label}{(' :: ' + str(extra)) if extra else ''}")

# ---------------------------------------------------------------- discovery
run = LepyRun.discover(DATA / "Napari_Lepy_Data")   # <- point at the INPUT folder
check("discovers execution folder from input folder",
      run.root.name == "Napari_Lepy_Data_Lepy_Execution_2026-08-07_13-52", run.root.name)
check("indexes all codes", len(run.codes) == 13, len(run.codes))
check("stats parsed (tab-delimited)", len(run.stats_by_code) == 12, len(run.stats_by_code))
check("header has 125 columns", len(run.header) == 125, len(run.header))
check("images found for every code", len(run.image_paths) == 13, len(run.image_paths))
check("morphology flags read", run.morph_error_codes == {"EcEs-Lep-00086"}, run.morph_error_codes)
check("error logs parsed (this run's are empty)",
      run.failure_reasons == {}, run.failure_reasons)
check("default output dir mirrors run name",
      run.default_output_dir.name.endswith("Napari_Corrections_2026-08-07_13-52"),
      run.default_output_dir.name)
print("  summary:", run.summary())
print("  batches of 5:", [len(b) for b in run.batches(5)])

# ---------------------------------------------------------------- record
rec = run.record("EcEs-Lep-00025")
check("has stats", rec.has_stats)
check("scale", abs(rec.scale - 166.0) < 1e-6, rec.scale)
check("size from stats", rec.size == (4000, 6000), rec.size)
poi = rec.poi_rowcol()
check("8 POIs as (row, col)", poi.shape == (8, 2), poi.shape)
check("body_top matches json", tuple(poi[0]) == (1503.0, 2941.0), tuple(poi[0]))
contour = rec.contour_px()
check("contour vertices", len(contour) == 2247, len(contour))
check("contour xmin/xmax match stats",
      contour[:, 0].min() == 1946 and contour[:, 0].max() == 3950,
      (contour[:, 0].min(), contour[:, 0].max()))
check("contour ymin/ymax match stats",
      contour[:, 1].min() == 1112 and contour[:, 1].max() == 2736,
      (contour[:, 1].min(), contour[:, 1].max()))

failed = run.record("EcEs-Lep-00462")
check("failed record detected", failed.failed and failed.poi_rowcol() is None)
# this run's error logs are empty (the .png was skipped by the reader's
# extension filter, not by a raised error) -- the reason must degrade to ""
check("missing failure reason degrades gracefully", failed.failure_reason == "",
      repr(failed.failure_reason))
check("failed record still has an image", failed.image_path is not None)
check("failed record has empty contour", len(failed.contour_px()) == 0)

# --------------------------------------------------- contour txt round-trip
text = contour_to_txt(contour, width=6000, height=4000)
tmp = Path(tempfile.gettempdir()) / "_contour_rt.txt"; tmp.write_text(text)
back = read_contour_txt(tmp, width=6000, height=4000)
check("contour txt round-trip", np.array_equal(contour, back),
      f"max diff {np.abs(contour - back).max() if contour.shape == back.shape else 'shape'}")

# --------------------------------------------------------------- recompute
base = rec.full_stats()
stats = rc.recompute(base, contour_px=contour, poi_rowcol=poi, shape=(4000, 6000),
                     scale=rec.scale, mask_changed=False)
for key in ("poi_dist_inner", "poi_dist_body", "poi_dist_inner_outer_l",
            "poi_dist_inner_outer_r", "poi_area_body", "poi_area_wing_l",
            "poi_area_wing_r", "poi_center_x", "poi_center_y"):
    same = abs(float(stats[key]) - float(base[key])) < 1e-6
    check(f"unchanged POIs reproduce {key}", same, f"{base[key]} -> {stats[key]}")

cstats = rc.contour_stats(contour, scale=rec.scale)
for key in ("contour_length", "contour_area", "contour_xmin", "contour_xmax",
            "contour_ymin", "contour_ymax", "contour_area_calibrated",
            "contour_width_calibrated", "contour_height_calibrated"):
    same = abs(float(cstats[key]) - float(base[key])) < 1e-3
    check(f"contour stats reproduce {key}", same, f"{base[key]} -> {cstats[key]}")

# moving a POI must change the traits
moved_poi = poi.copy()
moved_poi[2] += [480, -720]          # outer_l, ~5 mm
stats2 = rc.recompute(base, contour_px=contour, poi_rowcol=moved_poi,
                      shape=(4000, 6000), scale=rec.scale, mask_changed=False)
check("moving outer_l changes left FWL",
      abs(stats2["poi_dist_inner_outer_l"] - base["poi_dist_inner_outer_l"]) > 0,
      f"{base['poi_dist_inner_outer_l']} -> {stats2['poi_dist_inner_outer_l']}")
check("moving outer_l leaves right FWL alone",
      stats2["poi_dist_inner_outer_r"] == base["poi_dist_inner_outer_r"])
n_moved, max_shift = rc.poi_shift(poi, moved_poi)
check("poi_shift counts one point", n_moved == 1, n_moved)
check("poi_shift measures distance", abs(max_shift - np.hypot(480, 720)) < 1e-6, max_shift)

# LEPY quantises POI coordinates to whole mm before measuring, so small moves
# are invisible in poi_dist_*; the *_exact columns must still respond.
small = poi.copy()
small[2] += [80, -120]               # ~0.9 mm
q = rc.recompute(base, contour_px=contour, poi_rowcol=small, shape=(4000, 6000),
                 scale=rec.scale, mask_changed=False)
check("LEPY quantisation confirmed: sub-mm move leaves poi_dist_* unchanged",
      q["poi_dist_inner_outer_l"] == base["poi_dist_inner_outer_l"],
      q["poi_dist_inner_outer_l"])
check("exact distance responds to the same move",
      abs(q["poi_dist_inner_outer_l_exact"]
          - rc.exact_distances(poi, scale=rec.scale)["poi_dist_inner_outer_l_exact"]) > 0.1,
      f"{rc.exact_distances(poi, rec.scale)['poi_dist_inner_outer_l_exact']:.3f}"
      f" -> {q['poi_dist_inner_outer_l_exact']:.3f}")
check("exact distance matches hand calculation",
      abs(rc.exact_distances(poi, rec.scale)["poi_dist_inner_outer_l_exact"]
          - np.hypot(*(poi[4] - poi[2])) / rec.scale) < 1e-9)
check("exact keys present for all four distances",
      sum(1 for k in q if k.endswith("_exact")) == 4,
      [k for k in q if k.endswith("_exact")])

# mask/contour edit path
simplified = rc.simplify_contour(contour, 6.0)
check("simplify reduces vertices", 3 < len(simplified) < len(contour), len(simplified))
mask = rc.rasterize_contour(contour, (4000, 6000))
check("rasterised mask non-empty", mask.sum() > 0, int(mask.sum()))
recontour = rc.contour_from_mask(mask)
check("mask -> contour round-trips bbox",
      abs(int(recontour[:, 0].min()) - 1946) <= 2 and abs(int(recontour[:, 0].max()) - 3950) <= 2,
      (recontour[:, 0].min(), recontour[:, 0].max()))

poly = rc.contour_to_polygon(contour)
check("polygon is (row, col)", np.array_equal(poly[:, 0], contour[:, 1]))
check("polygon round-trip", np.array_equal(rc.polygon_to_contour(poly), contour))

# colour stats path (uses the real image -> slowest check)
rgb = rec.image()
check("image decodes", rgb is not None and rgb.shape == (4000, 6000, 3),
      None if rgb is None else rgb.shape)
col = rc.color_stats(rgb, mask)
for key in ("red_median", "green_median", "blue_median", "hue_median",
            "saturation_median", "luminance_median"):
    same = abs(float(col[key]) - float(base[key])) < 0.51
    check(f"colour stats reproduce {key}", same, f"{base[key]} -> {col[key]}")

# ------------------------------------------------------------------ status
db = ReviewStatusDB(OUT / "review_status.csv", source_run=run.name)
for code in run.codes:
    if code not in run.stats_by_code:
        db.set_status(code, FAILED)
db.set_status("EcEs-Lep-00025", OK, batch=0)
db.set_status("EcEs-Lep-00034", CORRECTED, batch=0, poi_corrected=True,
              mask_corrected=True, needs_reanalysis=True, n_poi_moved=2,
              max_poi_shift_px=14.5)
db.save()
db2 = ReviewStatusDB(OUT / "review_status.csv")
check("status db round-trips", db2.status("EcEs-Lep-00025") == OK, db2.status("EcEs-Lep-00025"))
e = db2.get("EcEs-Lep-00034")
check("bools survive", e.poi_corrected is True and e.mask_corrected is True, e)
check("ints survive", e.n_poi_moved == 2, e.n_poi_moved)
check("floats survive", abs(e.max_poi_shift_px - 14.5) < 1e-9, e.max_poi_shift_px)
check("corrected_codes", db2.corrected_codes() == ["EcEs-Lep-00034"], db2.corrected_codes())
check("reanalysis_codes", db2.reanalysis_codes() == ["EcEs-Lep-00034"], db2.reanalysis_codes())
check("progress excludes failed specimens",
      db2.progress(run.codes) == (2, 13), db2.progress(run.codes))

# ------------------------------------------------------------------ export
writer = CorrectionWriter(OUT, run, db2)
check("header taken from run", writer.header == run.header)
check("default_header matches run header", default_header() == run.header,
      f"{len(default_header())} vs {len(run.header)}")

writer.write_specimen("EcEs-Lep-00034", stats2, contour_px=simplified,
                      shape=(4000, 6000), mask=mask)
written = writer.write_tables({"EcEs-Lep-00034": stats2})
for name in ("stats_corrected.csv", "stats_combined_merged.csv",
             "corrected_codes.txt", "reanalysis_codes.txt", "review_status.csv",
             "stats_json/EcEs-Lep-00034.json", "contours_txt/EcEs-Lep-00034.txt",
             "masks_png/EcEs-Lep-00034.png"):
    check(f"wrote {name}", (OUT / name).exists())

import csv
with open(OUT / "stats_combined_merged.csv", newline="") as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))
check("merged table keeps every code with stats", len(rows) == 12, len(rows))
merged = {r["Code"]: r for r in rows}
check("merged row substituted",
      float(merged["EcEs-Lep-00034"]["poi_dist_inner_outer_l"]) == stats2["poi_dist_inner_outer_l"],
      merged["EcEs-Lep-00034"]["poi_dist_inner_outer_l"])
check("merged row untouched elsewhere",
      merged["EcEs-Lep-00025"]["poi_body_top_x"] == run.stats_by_code["EcEs-Lep-00025"]["poi_body_top_x"])
check("column order preserved",
      list(rows[0].keys()) == run.header, list(rows[0].keys())[:3])

reloaded = writer.load_corrected_stats()
check("corrected stats reload", set(reloaded) == {"EcEs-Lep-00034"}, set(reloaded))
check("reloaded values usable",
      rc.rowcol_from_stats(reloaded["EcEs-Lep-00034"]) is not None)

# ----------------------------------------------------------------- palette
for scheme in pal.SCHEMES:
    check(f"palette {scheme}: 8 colours",
          len(pal.colors(scheme)) == 8 and len(pal.symbols(scheme)) == 8)
check("default scheme has 8 distinct hues",
      len(set(pal.colors("role_symbol"))) == 8)

# ------------------------------------------- error-log parsing (synthetic)
with tempfile.TemporaryDirectory() as tmpdir:
    fake = Path(tmpdir) / "run_Lepy_Execution_2026-01-01_00-00"
    (fake / "errors_log").mkdir(parents=True)
    (fake / "errors_log" / "errors01.log").write_text(
        '[2026-01-01 00:00:00] Failed to process "/data/AB-Lep-00001.jpg". '
        'Reason (ValueError): no contour found\n'
        '[2026-01-01 00:00:01] Failed to process "/data/AB-Lep-00002.jpg". '
        'Reason (IndexError): empty mask\n')
    (fake / "morphological_errors_overview.txt").write_text(
        "Found errors\n===========\npoi_dist_inner: AB-Lep-00001, AB-Lep-00003\n"
        "contour_area: AB-Lep-00002\n")
    fake_run = LepyRun.discover(fake)
    check("error log: two failures parsed", len(fake_run.failure_reasons) == 2,
          fake_run.failure_reasons)
    check("error log: reason text kept",
          fake_run.failure_reasons.get("AB-Lep-00001") == "ValueError: no contour found",
          fake_run.failure_reasons.get("AB-Lep-00001"))
    check("morph overview: multi-code line split",
          fake_run.morph_error_codes == {"AB-Lep-00001", "AB-Lep-00002", "AB-Lep-00003"},
          fake_run.morph_error_codes)
    check("morph overview: reason recorded",
          fake_run.morph_error_reasons.get("AB-Lep-00003") == "poi_dist_inner",
          fake_run.morph_error_reasons)
    check("run with no stats_combined still indexes",
          fake_run.codes == [], fake_run.codes)

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
