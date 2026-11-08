"""Choose tau_ACM and the image-level decision rule on IDRiD *training* images.

    python tune_tau.py

Uses 15 IDRiD segmentation-training images (all with exudates; pixel F1 and
image sensitivity) and 25 healthy IDRiD grading-training images (retinopathy
grade 0, macular-edema risk 0 -> no exudates; image specificity).
DIARETDB1 is not used, so its score stays an independent test.
"""
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from exudates import Params, detect_exudates
from exudates.evaluation import load_gt_mask, load_rgb, pixel_metrics, summarize, lesion_metrics
from exudates.preprocessing import preprocess

SEG = Path("data/IDRiD/A. Segmentation")
GRD = Path("data/IDRiD/B. Disease Grading")
TAUS = [20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
MIN_AREAS = [25, 100, 250, 500, 1000, 2000]


def main():
    seg_files = sorted((SEG / "1. Original Images/a. Training Set").glob("*.jpg"))[::3][:15]
    rows = list(csv.reader(open(GRD / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")))
    healthy = [r[0] for r in rows[1:] if r and r[0] and r[1] == "0" and r[2] == "0"][::5][:25]

    results = {t: {"pos_px": [], "neg_px": [], "pix": []} for t in TAUS}
    for f in seg_files:
        pre = preprocess(load_rgb(f))
        gt = load_gt_mask(SEG / "2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates" / f"{f.stem}_EX.tif", pre) & pre.roi
        for t in TAUS:
            res = detect_exudates(None, replace(Params(), tau_acm=t), pre=pre)
            results[t]["pos_px"].append(int(res.mask.sum()))
            results[t]["pix"].append({**pixel_metrics(res.mask, gt, pre.roi), **lesion_metrics(res.mask, gt)})
        print("positive", f.stem, [results[t]["pos_px"][-1] for t in TAUS], flush=True)
    for name in healthy:
        pre = preprocess(load_rgb(GRD / "1. Original Images/a. Training Set" / f"{name}.jpg"))
        for t in TAUS:
            res = detect_exudates(None, replace(Params(), tau_acm=t), pre=pre)
            results[t]["neg_px"].append(int(res.mask.sum()))
        print("healthy ", name, [results[t]["neg_px"][-1] for t in TAUS], flush=True)

    print(f"\n{'tau':>5} {'pixF1':>6} {'minArea':>8} {'SE':>5} {'SP':>5} {'balAcc':>7}")
    best = None
    for t in TAUS:
        f1 = summarize(results[t]["pix"])["pixel_f1"]
        pos, neg = np.array(results[t]["pos_px"]), np.array(results[t]["neg_px"])
        for a in MIN_AREAS:
            se, sp = np.mean(pos >= a), np.mean(neg < a)
            bal = (se + sp) / 2
            score = bal + f1
            print(f"{t:5.0f} {f1:6.3f} {a:8d} {se:5.2f} {sp:5.2f} {bal:7.3f}")
            if best is None or score > best[0]:
                best = (score, t, a, f1, se, sp)
    print(f"\nBest (balanced accuracy + pixel F1): tau_ACM={best[1]}, min_area={best[2]} "
          f"(pixel F1 {best[3]:.3f}, SE {best[4]:.2f}, SP {best[5]:.2f})")


if __name__ == "__main__":
    main()
