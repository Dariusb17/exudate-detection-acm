"""Stage 3 - Confounding structures (Section III-A-3, Algorithm 1 lines 10-14).

  * ROI border: a thin ring along the edge of the retina, where the bright
    rim / dark corners produce strong false color gradients.
  * Blood vessels: morphological closing of the green channel with a disk
    1.25x as wide as the largest vessel fills the vessel lumens; the vessels
    are where the closing changed the image (closing - green).
  * Optic disc (OD): entropy-and-shape heuristic. Median filter + CLAHE,
    local entropy map, threshold, then the large connected component with
    the highest circularity is the OD (see the note in the function).

The paper gives the method but not every numeric setting; values that are not
in the paper are marked "(not in paper)" and expressed relative to the ROI
diameter D so they work at any resolution.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy

from .preprocessing import disk

LARGEST_VESSEL_FRAC = 0.012   # largest vessel width ~1.2 % of D (~18 px at D=1500) (not in paper)
VESSEL_SE_FACTOR = 1.25       # SE = 1.25 x largest vessel width (paper)
BORDER_FRAC = 0.02            # width of the removed ROI border ring (not in paper)
OD_DIAMETER_FRAC = 0.15       # expected OD diameter ~15 % of D (not in paper)
OD_ENTROPY_RADIUS_FRAC = 0.053  # local entropy window radius (not in paper)
OD_ENTROPY_PERCENTILE = 95      # keep the top 5 % entropy values (not in paper)


def roi_border(roi: np.ndarray, d: int) -> np.ndarray:
    # shrink the retina mask; the ring that disappears is the border
    inner = cv2.erode(roi.astype(np.uint8), disk(d * BORDER_FRAC)).astype(bool)
    return roi & ~inner


def detect_vessels(green: np.ndarray, roi: np.ndarray, d: int) -> np.ndarray:
    """Closing with SE = 1.25 x largest vessel width; vessels = (closing - green) > Otsu."""
    se = disk(VESSEL_SE_FACTOR * LARGEST_VESSEL_FRAC * d / 2)    # disk 1.25 x the largest vessel width
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, se)        # closing fills the dark vessels
    diff = cv2.subtract(closed, green)                           # large where a vessel was filled
    diff = cv2.medianBlur(diff, 5)                               # remove pixel noise
    inner = cv2.erode(roi.astype(np.uint8), disk(d * BORDER_FRAC)).astype(bool)
    thr = threshold_otsu(diff[inner])                            # automatic threshold, away from the border
    vessels = (diff > thr) & roi
    # vessels form long connected trees: drop small isolated blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vessels.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= (0.02 * d) ** 2
    return keep[labels]


def _circularity(area, perimeter):
    return 4 * np.pi * area / max(perimeter, 1e-6) ** 2


def detect_optic_disc_entropy(rgb: np.ndarray, roi: np.ndarray, d: int, low: bool = False):
    """Paper's OD heuristic: median + CLAHE -> local entropy -> threshold ->
    pick the large component with the highest circularity.

    Deviation: the paper keeps *low*-entropy candidates. On IDRiD the low-entropy
    area is the smooth macula (0/81 correct), while the OD - bright disc crossed
    by dark vessels - is the highest-entropy spot, so the top entropy values are
    kept instead (75/81 correct). Window radius and percentile were chosen on
    the IDRiD OD masks (not in paper).

    low=True keeps the low-entropy candidates instead, exactly as the paper
    states (used by the strict-paper mode, for comparison).

    Works on a 4x downscaled copy (entropy over large windows is slow).
    Returns (mask, center).
    """
    f = 4
    # work on a 4x smaller copy
    small = cv2.resize(rgb, (rgb.shape[1] // f, rgb.shape[0] // f), interpolation=cv2.INTER_AREA)
    roi_s = cv2.resize(roi.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    ds = d / f
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)                                               # median filter
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)       # contrast enhancement
    # local entropy: how varied the gray levels are around each pixel
    ent = entropy(gray, disk(ds * OD_ENTROPY_RADIUS_FRAC), mask=roi_s.astype(np.uint8)).astype(np.float32)
    inner = cv2.erode(roi_s.astype(np.uint8), disk(ds * 0.05)).astype(bool)
    if low:      # the paper as written: the 5 % lowest entropy values
        thr = np.percentile(ent[inner], 100 - OD_ENTROPY_PERCENTILE)
        cand = ((ent < thr) & inner).astype(np.uint8)
    else:        # our version: the 5 % highest entropy values
        thr = np.percentile(ent[inner], OD_ENTROPY_PERCENTILE)
        cand = ((ent > thr) & inner).astype(np.uint8)

    od_area = np.pi * (OD_DIAMETER_FRAC * ds / 2) ** 2      # expected area of the optic disc
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.1 * od_area:           # too small to be the OD
            continue
        circ = _circularity(area, cv2.arcLength(c, True))
        score = circ * min(area, od_area)   # "largest component with highest circularity"
        if score > best_score:
            best, best_score = c, score
    mask = np.zeros(rgb.shape[:2], np.uint8)
    if best is None:
        return mask.astype(bool), None
    # OD mask = circle around the chosen blob, back at full size
    (cx, cy), _ = cv2.minEnclosingCircle(best)
    center = (int(cx * f), int(cy * f))
    cv2.circle(mask, center, int(0.65 * OD_DIAMETER_FRAC * d), 1, -1)
    return mask.astype(bool), center


def detect_optic_disc_brightness(rgb: np.ndarray, roi: np.ndarray, vessels: np.ndarray, d: int):
    """Alternative OD detector (our own, not from the paper): among the
    brightest regions, the one with the most thick vessels. Returns (mask, center)."""
    r_od = d * OD_DIAMETER_FRAC / 2                        # expected OD radius
    inner = cv2.erode(roi.astype(np.uint8), disk(d * BORDER_FRAC)).astype(bool)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    L = cv2.morphologyEx(lab[..., 0], cv2.MORPH_CLOSE, disk(d * 0.012))      # lightness with vessels filled
    thick = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_OPEN, disk(d * 0.003)).astype(np.float32)

    def blur(img, sigma):  # masked Gaussian blur at reduced size
        f = 8
        size = (img.shape[1] // f, img.shape[0] // f)
        m = cv2.resize(inner.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        x = cv2.resize(img * inner, size, interpolation=cv2.INTER_AREA)
        out = cv2.GaussianBlur(x, (0, 0), sigma / f) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma / f), 1e-6)
        return cv2.resize(out, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)

    bright, density = blur(L, r_od * 0.6), blur(thick, r_od)    # smoothed brightness and vessel density
    v = bright[inner]
    bright = (bright - v.min()) / (v.max() - v.min() + 1e-6)
    bright[~inner] = -1
    peaks = (bright == ndi.maximum_filter(bright, size=int(2 * r_od) | 1)) & (bright >= 0.8) & inner
    ys, xs = np.nonzero(peaks)
    k = int(np.argmax(density[ys, xs]))                         # bright peak with the most vessels
    center = (int(xs[k]), int(ys[k]))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(mask, center, int(1.3 * r_od), 1, -1)
    return mask.astype(bool), center
