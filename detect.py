"""Run exudate detection on one image or a folder of images.

    python detect.py path/to/image.jpg
    python detect.py path/to/folder --out results/my_run --steps

For each image it saves <name>_overlay.png (exudates in blue), <name>_mask.png
and, with --steps, <name>_steps.png showing every stage of the pipeline.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from exudates import Params, detect_exudates, overlay
from exudates.evaluation import load_rgb, save_rgb
from exudates.visualize import step_figure

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--out", default="results/detect", help="output folder")
    ap.add_argument("--steps", action="store_true", help="also save the step-by-step figure")
    ap.add_argument("--paper-values", action="store_true", help="use the paper's tau_ACM = 20 instead of the tuned value")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    for f in files:
        t = time.time()
        res = detect_exudates(load_rgb(f), Params.paper() if args.paper_values else Params())
        save_rgb(out / f"{f.stem}_overlay.png", np.hstack([res.pre.rgb, overlay(res.pre.rgb, res.mask)]))
        save_rgb(out / f"{f.stem}_mask.png", np.dstack([res.mask.astype(np.uint8) * 255] * 3))
        if args.steps:
            step_figure(res, out / f"{f.stem}_steps.png", title=f.name)
        n = int(res.mask.sum())
        print(f"{f.name}: {n} exudate pixels  ({time.time() - t:.1f}s)")
    print(f"Results saved in {out.resolve()}")


if __name__ == "__main__":
    main()
