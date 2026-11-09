"""Figure with every intermediate step of Algorithm 1 (useful for the report)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detection import Result, overlay


def _tint(rgb, mask, color, alpha=0.6):
    out = rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def step_panels(res: Result, gt: np.ndarray | None = None) -> list:
    """(title, image, colormap or None) for every stage of Algorithm 1."""
    s, pre = res.steps, res.pre
    rgb = pre.rgb
    anat = _tint(rgb, s["od"], (255, 0, 255))
    anat = _tint(anat, s["border"], (0, 255, 255))
    anat[s["vessels"]] = (0, 255, 0)
    dist = s["distance"].copy()
    dist[~pre.roi] = 0

    panels = [
        ("Input", rgb, None),
        ("1. Histogram matching (L*a*b*, ROI)", pre.rgb_norm, None),
        ("2. CIEDE2000 distance to C_ref", np.minimum(dist, 60), "viridis_r"),
        ("2. ACM gradient G (Eq. 19)", s["gradient"], "magma"),
        ("2. G > 0.15", s["b_cand"], "gray"),
        ("2. Filled candidates", s["b_filled"], "gray"),
        ("3. OD (magenta), vessels (green), border (cyan)", anat, None),
        ("3. Clean candidates", s["b_clean"], "gray"),
        ("4. Green - reconstruction", np.where(s["b_clean"], s["recon_diff"], 0), "magma"),
        ("4. Thresholded (E_init)", s["e_init"], "gray"),
        ("4. Edge-density filtered", s["e_refined"], "gray"),
        ("Final (5x5 dilation)", overlay(rgb, res.mask), None),
    ]
    if gt is not None:
        panels.append(("vs. ground truth: green=TP  yellow=FP  red=FN", compare_image(rgb, res.mask, gt), None))
    return panels


def compare_image(rgb, mask, gt):
    out = rgb.copy()
    out[gt & ~mask] = (255, 0, 0)
    out[mask & ~gt] = (255, 255, 0)
    out[mask & gt] = (0, 255, 0)
    return out


def step_figure(res: Result, path, gt: np.ndarray | None = None, title: str = "") -> None:
    panels = step_panels(res, gt)
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, img, cmap) in zip(axes.ravel(), panels):
        if cmap:
            img = img.astype(np.float32)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=max(1e-6, float(np.percentile(img, 99.8))))
        else:
            ax.imshow(img)
        ax.set_title(name, fontsize=11)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
