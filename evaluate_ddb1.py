"""Image-level exudate detection on DIARETDB1 - the evaluation of the paper (Table 3).

    python evaluate_ddb1.py
    python evaluate_ddb1.py --steps --limit 10

An image is predicted "with exudates" when the detector marks at least
--min-area pixels. Ground truth: DIARETDB1 hard-exudate maps store the experts'
mean confidence; following the DIARETDB1 protocol, an image counts as
"with exudates" when some area has confidence >= 0.75.
Reports sensitivity, specificity and accuracy over all 89 images, plus
pixel-level numbers on the confident (>= 0.75) areas.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from exudates import Params, detect_exudates, overlay
from exudates.evaluation import load_rgb, save_rgb
from exudates.visualize import step_figure

ROOT = "data/DIARETDB1/diaretdb1_v_1_1/resources/images"
CONF_LEVEL = 0.75


def load_gt(path, pre, level=CONF_LEVEL):
    """Confidence map (0..1) -> binary mask at the working geometry."""
    g = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 252.0
    y0, y1, x0, x1 = pre.crop_box
    g = cv2.resize(g[y0:y1, x0:x1], pre.roi.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return (g >= level - 1e-3) & pre.roi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT)
    ap.add_argument("--min-area", type=int, default=None, help="pixels needed to call an image positive (default: Params.image_min_area)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--out", default="results/eval_ddb1")
    ap.add_argument("--paper-values", action="store_true", help="use the paper's tau_ACM = 20 instead of the tuned value")
    args = ap.parse_args()

    root = Path(args.data)
    images = sorted((root / "ddb1_fundusimages").glob("image*.png"))
    if args.limit:
        images = images[: args.limit]
    out = Path(args.out)
    params = Params.paper() if args.paper_values else Params()
    min_area = args.min_area if args.min_area is not None else params.image_min_area
    rows = []
    for f in images:
        res = detect_exudates(load_rgb(f), params)
        gt = load_gt(root / "ddb1_groundtruth" / "hardexudates" / f.name, res.pre)
        has_gt = bool(gt.any())
        detected = int(res.mask.sum())
        pred = detected >= min_area
        tp = int((res.mask & gt).sum()); fp = int((res.mask & ~gt).sum()); fn = int((~res.mask & gt).sum())
        rows.append(dict(image=f.stem, gt_exudates=int(has_gt), predicted=int(pred), detected_px=detected,
                         tp=tp, fp=fp, fn=fn))
        flag = "OK " if pred == has_gt else ("FN " if has_gt else "FP ")
        print(f"{flag}{f.stem}: ground truth {'EX' if has_gt else '--'}  detected {detected} px")
        save_rgb(out / f"{f.stem}_overlay.jpg", np.hstack([res.pre.rgb, overlay(res.pre.rgb, res.mask)]))
        if args.steps:
            step_figure(res, out / f"{f.stem}_steps.jpg", gt=gt, title=f.stem)

    gt_v = np.array([r["gt_exudates"] for r in rows], bool)
    pr_v = np.array([r["predicted"] for r in rows], bool)
    TP, TN = int(np.sum(gt_v & pr_v)), int(np.sum(~gt_v & ~pr_v))
    FP, FN = int(np.sum(~gt_v & pr_v)), int(np.sum(gt_v & ~pr_v))
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    summary = dict(images=len(rows), with_exudates=int(gt_v.sum()),
                   TP=TP, TN=TN, FP=FP, FN=FN,
                   sensitivity=TP / max(TP + FN, 1), specificity=TN / max(TN + FP, 1),
                   accuracy=(TP + TN) / max(len(rows), 1),
                   pixel_sensitivity=tp / max(tp + fn, 1), pixel_ppv=tp / max(tp + fp, 1))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== DIARETDB1 image-level ({summary['images']} images, {summary['with_exudates']} with exudates) ===")
    print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")
    for k in ("sensitivity", "specificity", "accuracy", "pixel_sensitivity", "pixel_ppv"):
        print(f"  {k:18s} {summary[k]:.3f}")
    # how the image-level result depends on the decision rule
    px = np.array([r["detected_px"] for r in rows])
    print(f"\n  min. area   SE    SP    Acc   (current rule: {min_area} px)")
    for a in (25, 250, 500, 1000, 2000, 4000):
        p_ = px >= a
        se, sp = np.mean(p_[gt_v]), np.mean(~p_[~gt_v])
        print(f"  {a:9d}  {se:.2f}  {sp:.2f}  {np.mean(p_ == gt_v):.2f}")


if __name__ == "__main__":
    main()
