"""Comparison against expert ground-truth masks (pixel- and lesion-level)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocessing import Preprocessed


def load_rgb(path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    buf.tofile(str(path))


def load_gt_mask(path, pre: Preprocessed) -> np.ndarray:
    """Load a ground-truth mask and bring it to the working geometry."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[..., :3].max(axis=2)   # some IDRiD masks carry an opaque alpha channel
    y0, y1, x0, x1 = pre.crop_box
    m = (img[y0:y1, x0:x1] > 0).astype(np.float32)
    h, w = pre.roi.shape
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return m > 0.25


def pixel_metrics(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> dict:
    p, g = pred & fov, gt & fov
    tp = int(np.sum(p & g)); fp = int(np.sum(p & ~g)); fn = int(np.sum(~p & g))
    return dict(tp=tp, fp=fp, fn=fn)


def lesion_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """A GT lesion counts as found if any detected pixel touches it, and a
    detected region counts as correct if it touches any GT lesion."""
    ng, lg = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    npd, lp = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    gt_hit = np.unique(lg[pred & (lg > 0)])
    pd_hit = np.unique(lp[gt & (lp > 0)])
    return dict(gt_lesions=ng - 1, gt_found=len(gt_hit),
                det_regions=npd - 1, det_correct=len(pd_hit))


def summarize(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    se = tp / max(tp + fn, 1); ppv = tp / max(tp + fp, 1)
    f1 = 2 * se * ppv / max(se + ppv, 1e-9)
    gl = sum(r["gt_lesions"] for r in rows); gf = sum(r["gt_found"] for r in rows)
    dr = sum(r["det_regions"] for r in rows); dc = sum(r["det_correct"] for r in rows)
    return dict(images=len(rows),
                pixel_sensitivity=se, pixel_ppv=ppv, pixel_f1=f1,
                lesion_sensitivity=gf / max(gl, 1), lesion_ppv=dc / max(dr, 1))
