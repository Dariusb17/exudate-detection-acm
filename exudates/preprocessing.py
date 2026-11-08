"""Stage 1 - Pre-processing (Section III-A-1, Algorithm 1 lines 1-3).

  1. Region of Interest (ROI): the circular retina area, black corners excluded
  2. resize to a standard field of view (FOV), as done for all datasets in the
     paper ("image resizing to a standard field of view", Sec. IV-A-3)
  3. RGB -> CIE L*a*b*
  4. histogram matching to a fixed reference image: the mean and standard
     deviation of each L*a*b* channel are matched to those of the reference
     (DIARETDB1 image006 in the paper), using only the ROI pixels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Standard FOV diameter in pixels. The paper fixes the MA width as
# w_max = D_ROI / 99 ~ 15 px, i.e. D_ROI ~ 1500 px.
WORK_WIDTH = 1500

# Reference statistics (per L*, a*, b* channel: mean, std over the ROI).
# Computed from DIARETDB1 diaretdb1_image006.png by `python make_reference.py`,
# which writes reference_stats.json next to this file.
REFERENCE_FILE = Path(__file__).with_name("reference_stats.json")


@dataclass
class Preprocessed:
    rgb: np.ndarray        # resized original, uint8 RGB
    lab: np.ndarray        # color-normalized image, float32 CIE L*a*b*
    rgb_norm: np.ndarray   # color-normalized image as uint8 RGB
    roi: np.ndarray        # bool mask of the retina
    crop_box: tuple        # (y0, y1, x0, x1) in original pixel coordinates
    scale: float           # working size / original size
    diameter: int          # ROI diameter in working pixels
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def green(self) -> np.ndarray:
        """Green channel of the normalized image (used by stages 3 and 4)."""
        return self.rgb_norm[..., 1]


def disk(radius: float) -> np.ndarray:
    radius = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def roi_mask(rgb: np.ndarray) -> np.ndarray:
    """Retina mask from the red channel (the corners of a fundus photo are ~black)."""
    red = cv2.GaussianBlur(rgb[..., 0], (5, 5), 0)
    thr = max(20, 0.08 * float(np.percentile(red, 99)))
    mask = (red > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> CIE L*a*b* (L in [0, 100], a/b signed), float32."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)


def crop_and_resize(rgb_full: np.ndarray, work_width: int = WORK_WIDTH):
    roi_full = roi_mask(rgb_full)
    ys, xs = np.where(roi_full)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb, roi = rgb_full[y0:y1, x0:x1], roi_full[y0:y1, x0:x1]
    scale = work_width / rgb.shape[1]
    size = (work_width, int(round(rgb.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, size, interpolation=interp)
    roi = cv2.resize(roi.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return rgb, roi, (y0, y1, x0, x1), scale


def lab_stats(lab: np.ndarray, roi: np.ndarray) -> dict:
    v = lab[roi]
    return {"mean": v.mean(axis=0).tolist(), "std": v.std(axis=0).tolist()}


def load_reference(path: Path = REFERENCE_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python make_reference.py` once to compute the "
            "color statistics of the reference image")
    return json.loads(path.read_text())


def histogram_matching(lab: np.ndarray, roi: np.ndarray, ref: dict) -> np.ndarray:
    """Match mean and std of every L*a*b* channel to the reference, on the ROI."""
    out = lab.copy()
    mu, sd = lab[roi].mean(axis=0), lab[roi].std(axis=0) + 1e-6
    ref_mu, ref_sd = np.array(ref["mean"]), np.array(ref["std"])
    out[roi] = (lab[roi] - mu) / sd * ref_sd + ref_mu
    out[~roi] = 0
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    return out.astype(np.float32)


def preprocess(rgb_full: np.ndarray, work_width: int = WORK_WIDTH, ref: dict | None = None) -> Preprocessed:
    rgb, roi, box, scale = crop_and_resize(rgb_full, work_width)
    lab = histogram_matching(rgb_to_lab(rgb), roi, ref or load_reference())
    rgb_norm = lab_to_rgb(lab)
    rgb_norm[~roi] = 0
    return Preprocessed(rgb, lab, rgb_norm, roi, box, scale, work_width)
