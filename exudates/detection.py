"""Exudate detection - Section III-A / Algorithm 1 of Rostamian & Shekari (2026).

Stage 1  Pre-processing: RGB -> L*a*b*, histogram matching to the reference (ROI)
Stage 2  Candidate extraction via Adaptive Color Morphology (ACM):
           G = ACM gradient (C_ref, tau_ACM)             Eq. (19)
           B_cand = G > tau_binarize (G normalized to [0, 1])
           B_filled = fill holes
Stage 3  Confounding structure elimination:
           B_clean = B_filled AND NOT (optic disc OR vessels OR ROI border)
Stage 4  Morphological reconstruction & refinement:
           marker = green - B_clean, mask = green
           recon  = reconstruction by dilation(marker, mask)
           E_init = (mask - recon) > tau_recon
           regions of E_init with sparse Sobel edge pixels are removed
           E = dilation(E_refined, 5x5)

Parameters from Table 1 / Algorithm 1 are marked (paper); the rest are not
given in the paper and were chosen on the IDRiD training images (tune.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import reconstruction

from . import anatomy
from .color_morphology import C_REF, TAU_ACM, AdaptiveColorMorphology, reference_distance
from .preprocessing import Preprocessed, disk, preprocess


@dataclass
class Params:
    c_ref: tuple = C_REF              # (paper) reference exudate color
    tau_acm: float = 28.0             # CIEDE2000 distance threshold (paper: 20, see README)
    tau_binarize: float = 0.15        # (paper) threshold on the normalized gradient
    g_norm: object = "max"            # how G is normalized: "max" or a fixed value (not in paper)
    acm_window: int = 5               # ACM structuring element, px (not in paper)
    cand_dilate: int = 4              # "filling candidate regions with dilation", px (not in paper)
    tau_recon: float = 10.0           # threshold on mask - reconstruction, gray levels (not in paper)
    tau_edge: float = 30.0            # Sobel magnitude threshold (not in paper)
    edge_density: float = 0.1         # min fraction of edge pixels on a region's outline (not in paper)
    edge_mode: str = "outline"        # where edges are counted: "outline", "area" or "pixel" (see below)
    final_dilate: int = 5             # (paper) final 5x5 dilation
    od_method: str = "entropy"        # "entropy" (highest entropy), "entropy_low" (paper as written), "brightness"
    image_min_area: int = 2000        # px marked to call an image "with exudates" (not in paper)

    @classmethod
    def paper(cls, **kw) -> "Params":
        """Strictly the paper's Table 1 values (tau_ACM = 20, G normalized by its
        maximum); settings the paper does not give keep their tuned values."""
        return cls(**{"tau_acm": TAU_ACM, "g_norm": "max", **kw})

    @classmethod
    def paper_strict(cls, **kw) -> "Params":
        """Algorithm 1 taken literally wherever the paper is specific:
        tau_ACM = 20, optic disc from the low-entropy candidates, pixel-wise
        E_init AND E_edges (line 21), no extra dilation of the candidates.
        Thresholds the paper does not give keep our values."""
        return cls(**{"tau_acm": TAU_ACM, "g_norm": "max", "od_method": "entropy_low",
                      "edge_mode": "pixel", "cand_dilate": 0, **kw})

    @classmethod
    def from_flags(cls, paper_values: bool = False, strict_paper: bool = False, **kw) -> "Params":
        """Parameters for the --paper-values / --strict-paper command-line options."""
        if strict_paper:
            return cls.paper_strict(**kw)
        return cls.paper(**kw) if paper_values else cls(**kw)


@dataclass
class Result:
    pre: Preprocessed
    mask: np.ndarray
    steps: dict = field(default_factory=dict)
    od_center: tuple | None = None


def _keep_edge_dense_regions(mask, edges, min_density, mode="outline"):
    """Remove regions that do not correspond to strong edges.

    The fraction of strong-edge pixels is measured per region and regions
    below min_density are removed. Where it is measured depends on `mode`:

      "outline"  on the region's outline only (a 3 px ring: 1 px outside,
                 2 px inside). Exudates have sharp borders, but a large plaque
                 is smooth inside, so its outline is what must be sharp.
      "area"     on the whole region (+ 1 px border). Our first version; it
                 removes large solid exudates, whose interior has no edges.
    """
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)   # number every region
    if n <= 1:
        return mask.copy()
    # each region grown by 1 px, so the edge just outside it counts too
    grown = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)
    if mode == "outline":
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        grown[core] = 0                                   # keep only the ring around each region
    total = np.bincount(grown.ravel(), minlength=n)                               # pixels per region
    edge = np.bincount(grown.ravel(), weights=edges.ravel().astype(float), minlength=n)   # edge pixels per region
    keep = edge / np.maximum(total, 1) >= min_density
    keep[0] = False
    return keep[labels]


def detect_exudates(rgb_full: np.ndarray | None, params: Params | None = None,
                    pre: Preprocessed | None = None) -> Result:
    p = params or Params()
    pre = pre or preprocess(rgb_full)                     # Stage 1: crop, resize, L*a*b*, color matching
    d, roi, green = pre.diameter, pre.roi, pre.green
    steps = {}

    # ---- Stage 2: candidate extraction via ACM ------------------------------
    # CIEDE2000 distance of every pixel to C_ref (kept in pre.cache so tuning can reuse it)
    dkey = ("dist", tuple(p.c_ref))
    if dkey not in pre.cache:
        pre.cache[dkey] = reference_distance(pre.lab, roi, p.c_ref)
    dist = pre.cache[dkey]
    # adaptive gradient G, Eq. (19)
    key = ("acm", tuple(p.c_ref), p.tau_acm, p.acm_window)
    if key not in pre.cache:
        acm = AdaptiveColorMorphology(pre.lab, roi, p.c_ref, p.tau_acm, dist=dist)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (p.acm_window, p.acm_window))
        pre.cache[key] = acm.gradient(se).astype(np.float16)
    G = pre.cache[key].astype(np.float32)
    G[~roi] = 0
    # G is scaled to [0, 1] before the 0.15 threshold. The paper does not say
    # how: "max" divides by the image's own maximum, a number divides by a
    # fixed CIEDE2000 value (same scale for every image).
    scale = float(G.max()) if p.g_norm == "max" else float(p.g_norm)
    G_norm = G / scale if scale > 0 else G
    b_cand = G_norm > p.tau_binarize                      # outlines of exudate-colored regions
    b_filled = ndi.binary_fill_holes(b_cand)              # outlines -> solid regions
    steps.update(distance=dist, gradient=G_norm, b_cand=b_cand, b_filled=b_filled)

    # ---- Stage 3: optic disc, vessels, ROI border ---------------------------
    akey = ("anatomy", p.od_method)
    if akey not in pre.cache:
        vessels = anatomy.detect_vessels(green, roi, d)
        if p.od_method in ("entropy", "entropy_low"):
            od, center = anatomy.detect_optic_disc_entropy(pre.rgb, roi, d, low=p.od_method == "entropy_low")
        else:
            od, center = anatomy.detect_optic_disc_brightness(pre.rgb, roi, vessels, d)
        pre.cache[akey] = (vessels, od, center, anatomy.roi_border(roi, d))
    vessels, od, od_center, border = pre.cache[akey]
    b_clean = b_filled & ~(od | vessels | border)         # Algorithm 1, line 14
    if p.cand_dilate > 0:                                 # grow candidates so they cover the whole lesion
        b_clean = cv2.dilate(b_clean.astype(np.uint8), disk(p.cand_dilate)).astype(bool)
        b_clean &= roi & ~od & ~border
    steps.update(vessels=vessels, od=od, border=border, b_clean=b_clean)

    # ---- Stage 4: morphological reconstruction & refinement ----------------
    mask_img = green.astype(np.float32)                   # mask = green channel
    marker = mask_img.copy()
    marker[b_clean] = 0                                   # marker = mask - candidates
    recon = reconstruction(marker, mask_img, method="dilation")   # background grows back into the candidates
    diff = mask_img - recon                               # how much brighter than the surroundings
    e_init = (diff > p.tau_recon) & b_clean

    # Sobel edge strength of the green channel
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.hypot(gx, gy)
    e_edges = sobel > p.tau_edge
    if p.edge_mode == "pixel":                            # Algorithm 1, line 21, taken literally
        e_refined = e_init & e_edges
    else:                                                 # the text: remove regions without strong edges
        e_refined = _keep_edge_dense_regions(e_init, e_edges, p.edge_density, p.edge_mode)

    k = p.final_dilate                                    # final 5x5 dilation
    e = cv2.dilate(e_refined.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) if k > 1 else e_refined
    e &= roi
    steps.update(recon_diff=diff, sobel=sobel, e_init=e_init, e_refined=e_refined)
    return Result(pre=pre, mask=e, steps=steps, od_center=od_center)


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Paint detected exudates in solid blue, like the figures in the paper."""
    out = rgb.copy()
    out[mask] = color
    return out
