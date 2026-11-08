"""Evaluate the detector on the IDRiD segmentation set (hard exudates).

    python evaluate.py --split test
    python evaluate.py --split train --limit 10 --steps

Expects the IDRiD "A. Segmentation" folder under data/IDRiD (see README).
Writes per-image metrics to results/eval_<split>/metrics.csv, overlays and,
with --steps, the step-by-step figures.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from exudates import Params, detect_exudates, overlay
from exudates.evaluation import (lesion_metrics, load_gt_mask, load_rgb, pixel_metrics,
                                 save_rgb, summarize)
from exudates.visualize import step_figure

SPLITS = {"train": "a. Training Set", "test": "b. Testing Set"}


def find_root(base: Path) -> Path:
    for cand in [base, *base.rglob("1. Original Images")]:
        root = cand.parent if cand.name == "1. Original Images" else cand
        if (root / "1. Original Images").is_dir():
            return root
    raise SystemExit(f"IDRiD 'A. Segmentation' folder not found under {base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/IDRiD")
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paper-values", action="store_true", help="use the paper's tau_ACM = 20 instead of the tuned value")
    args = ap.parse_args()

    root = find_root(Path(args.data))
    img_dir = root / "1. Original Images" / SPLITS[args.split]
    gt_dir = root / "2. All Segmentation Groundtruths" / SPLITS[args.split] / "3. Hard Exudates"
    out = Path(args.out or f"results/eval_{args.split}")
    images = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]

    params = Params.paper() if args.paper_values else Params()
    rows = []
    for f in images:
        gt_path = gt_dir / f"{f.stem}_EX.tif"
        if not gt_path.exists():
            continue
        t = time.time()
        res = detect_exudates(load_rgb(f), params)
        gt = load_gt_mask(gt_path, res.pre) & res.pre.roi
        row = dict(image=f.stem, **pixel_metrics(res.mask, gt, res.pre.roi), **lesion_metrics(res.mask, gt))
        rows.append(row)
        s = summarize([row])
        print(f"{f.stem}: pixel SE={s['pixel_sensitivity']:.2f} PPV={s['pixel_ppv']:.2f} "
              f"F1={s['pixel_f1']:.2f} | lesions found {row['gt_found']}/{row['gt_lesions']} "
              f"({time.time() - t:.1f}s)")
        save_rgb(out / f"{f.stem}_overlay.jpg", np.hstack([res.pre.rgb, overlay(res.pre.rgb, res.mask)]))
        if args.steps:
            step_figure(res, out / f"{f.stem}_steps.jpg", gt=gt, title=f.stem)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ({} images, {} set) ===".format(summary["images"], args.split))
    for k, v in summary.items():
        if k != "images":
            print(f"  {k:20s} {v:.3f}")


if __name__ == "__main__":
    main()
