"""Compute the color statistics of the histogram-matching reference image.

    python make_reference.py                       # DIARETDB1 image006 (as in the paper)
    python make_reference.py path/to/other.png     # any other reference

Writes exudates/reference_stats.json (mean and std of L*, a*, b* inside the ROI,
after resizing to the standard FOV). Only needs to be run once.
"""
import json
import sys
from pathlib import Path

from exudates.evaluation import load_rgb
from exudates.preprocessing import REFERENCE_FILE, crop_and_resize, lab_stats, rgb_to_lab

DEFAULT = "data/DIARETDB1/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image006.png"


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    rgb, roi, _, _ = crop_and_resize(load_rgb(path))
    stats = lab_stats(rgb_to_lab(rgb), roi)
    stats["source"] = path.name
    REFERENCE_FILE.write_text(json.dumps(stats, indent=2))
    print(f"Reference statistics from {path}:")
    for ch, m, s in zip("Lab", stats["mean"], stats["std"]):
        print(f"  {ch}*: mean {m:7.2f}  std {s:6.2f}")
    print(f"saved to {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
