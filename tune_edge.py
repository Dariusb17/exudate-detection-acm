"""Choose the edge-filter settings on IDRiD *training* images.

    python tune_edge.py

The paper removes "regions that do not correspond to strong edges" but gives
neither the Sobel threshold nor how many edge pixels a region needs. This
script tries a grid of both, for the two ways of counting edge pixels
("outline" and "area", see detection._keep_edge_dense_regions), on the same
images as tune_tau.py: 15 IDRiD segmentation-training images (pixel F1) and
25 healthy IDRiD grading-training images (image-level specificity).
IDRiD test and DIARETDB1 are not used.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

from exudates import Params, detect_exudates
from exudates.detection import _keep_edge_dense_regions
from exudates.evaluation import load_gt_mask, load_rgb

SEG = Path("data/IDRiD/A. Segmentation")
GRD = Path("data/IDRiD/B. Disease Grading")
TAU_EDGES = [15.0, 20.0, 25.0, 30.0, 40.0]
DENSITIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
MIN_AREAS = [250, 500, 1000, 2000, 4000, 8000]


def final_mask(steps, roi, mode, tau_edge, density, k=5):
    """Stage 4 from E_init on: edge filter + final dilation."""
    refined = _keep_edge_dense_regions(steps["e_init"], steps["sobel"] > tau_edge, density, mode)
    return cv2.dilate(refined.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & roi


def main():
    seg_files = sorted((SEG / "1. Original Images/a. Training Set").glob("*.jpg"))[::3][:15]
    rows = list(csv.reader(open(GRD / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")))
    healthy = [r[0] for r in rows[1:] if r and r[0] and r[1] == "0" and r[2] == "0"][::5][:25]
    combos = [(m, t, d) for m in ("outline", "area") for t in TAU_EDGES for d in DENSITIES]
    counts = {c: np.zeros(3) for c in combos}            # tp, fp, fn over all positive images
    pos_px = {c: [] for c in combos}
    neg_px = {c: [] for c in combos}

    for i, f in enumerate(seg_files):
        res = detect_exudates(load_rgb(f), Params())       # stages 1-3 and E_init are the same for every combo
        gt_file = SEG / "2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates" / f"{f.stem}_EX.tif"
        gt = load_gt_mask(gt_file, res.pre) & res.pre.roi
        for c in combos:
            m = final_mask(res.steps, res.pre.roi, *c)
            counts[c] += ((m & gt).sum(), (m & ~gt).sum(), (~m & gt).sum())
            pos_px[c].append(int(m.sum()))
        print(f"with exudates {i + 1}/{len(seg_files)}: {f.stem}", flush=True)
    for i, name in enumerate(healthy):
        res = detect_exudates(load_rgb(GRD / "1. Original Images/a. Training Set" / f"{name}.jpg"), Params())
        for c in combos:
            neg_px[c].append(int(final_mask(res.steps, res.pre.roi, *c).sum()))
        print(f"healthy {i + 1}/{len(healthy)}: {name}", flush=True)

    print(f"\n{'mode':>8} {'tauE':>5} {'dens':>5} {'pixSE':>6} {'pixPPV':>6} {'pixF1':>6} {'minArea':>8} {'SE':>5} {'SP':>5} {'score':>6}")
    results = []
    for c in combos:
        tp, fp, fn = counts[c]
        se_p, ppv, f1 = tp / max(tp + fn, 1), tp / max(tp + fp, 1), 2 * tp / max(2 * tp + fp + fn, 1)
        pos, neg = np.array(pos_px[c]), np.array(neg_px[c])
        # best image rule for this combo (balanced accuracy)
        bal, a = max(((np.mean(pos >= a) + np.mean(neg < a)) / 2, a) for a in MIN_AREAS)
        results.append((bal + f1, c, se_p, ppv, f1, a, np.mean(pos >= a), np.mean(neg < a)))
    for score, (m, t, d), se_p, ppv, f1, a, se, sp in sorted(results, reverse=True)[:15]:
        print(f"{m:>8} {t:5.0f} {d:5.2f} {se_p:6.3f} {ppv:6.3f} {f1:6.3f} {a:8d} {se:5.2f} {sp:5.2f} {score:6.3f}")
    for score, (m, t, d), se_p, ppv, f1, a, se, sp in results:
        if (m, t, d) == ("area", 30.0, 0.3):
            print(f"\nFirst version (area, 30, 0.30): pixel SE {se_p:.3f} PPV {ppv:.3f} F1 {f1:.3f}; "
                  f"min_area {a}: SE {se:.2f} SP {sp:.2f}; score {score:.3f}")
    score, (m, t, d), se_p, ppv, f1, a, se, sp = max(results)
    print(f"Best (balanced accuracy + pixel F1): edge_mode={m}, tau_edge={t}, edge_density={d}, min_area={a}")


if __name__ == "__main__":
    main()
