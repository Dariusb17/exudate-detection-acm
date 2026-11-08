"""Check optic disc localization against the IDRiD optic-disc masks.

    python evaluate_od.py                  # both methods
    python evaluate_od.py --method entropy

A detection is correct when the detected OD center falls inside the expert mask.
"""
import argparse
from pathlib import Path

from exudates import anatomy
from exudates.evaluation import load_gt_mask, load_rgb
from exudates.preprocessing import preprocess
from evaluate import SPLITS, find_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/IDRiD")
    ap.add_argument("--method", choices=["entropy", "brightness", "both"], default="both")
    args = ap.parse_args()
    methods = ["entropy", "brightness"] if args.method == "both" else [args.method]
    root = find_root(Path(args.data))
    hits = {m: [] for m in methods}
    misses = {m: [] for m in methods}
    for split in SPLITS.values():
        for f in sorted((root / "1. Original Images" / split).glob("*.jpg")):
            pre = preprocess(load_rgb(f))
            gt = load_gt_mask(root / "2. All Segmentation Groundtruths" / split / "5. Optic Disc" / f"{f.stem}_OD.tif", pre)
            vessels = anatomy.detect_vessels(pre.green, pre.roi, pre.diameter)
            for m in methods:
                if m == "entropy":
                    _, c = anatomy.detect_optic_disc_entropy(pre.rgb, pre.roi, pre.diameter)
                else:
                    _, c = anatomy.detect_optic_disc_brightness(pre.rgb, pre.roi, vessels, pre.diameter)
                ok = c is not None and gt[c[1], c[0]]
                (hits if ok else misses)[m].append(f.stem)
    for m in methods:
        n = len(hits[m]) + len(misses[m])
        print(f"[{m}] optic disc located correctly in {len(hits[m])}/{n} images ({100 * len(hits[m]) / n:.1f}%)")
        if misses[m]:
            print("   missed:", ", ".join(misses[m]))


if __name__ == "__main__":
    main()
