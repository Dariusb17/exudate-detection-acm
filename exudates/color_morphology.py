"""Adaptive Color Morphology (ACM), Section III-A-2, Eqs. (17)-(19).

Every pixel is judged by its CIEDE2000 distance to a reference exudate color
C_ref = (85, 10, 40) in CIE L*a*b* (light yellow). With a threshold tau = 20:

  adaptive dilation (Eq. 17):
      inside the window W_p, take the pixel q whose color is CLOSEST to C_ref;
      if that distance is < tau, the output is C_q, else C_p stays.
      -> exudate-like colors spread outwards.

  adaptive erosion (Eq. 18):
      only if the centre pixel itself is exudate-like (distance < tau): output
      the color of the pixel in W_p that is FARTHEST from C_ref, else C_p stays.
      -> exudate-like regions shrink, everything else is left untouched.

  adaptive gradient (Eq. 19):
      G = CIEDE2000(dilation, erosion), large on the borders of exudate-like
      regions and ~0 elsewhere.

The output colors are always colors that exist in the image (vector
morphology: no false colors are created).

Implementation: the arg-min / arg-max inside a window is found with ordinary
grayscale erosion / dilation applied to the *rank* of each pixel's distance
(rank -> pixel index lookup gives back the winning pixel and its color).
"""
from __future__ import annotations

import cv2
import numpy as np

from .ciede2000 import delta_e00

C_REF = (85.0, 10.0, 40.0)   # reference exudate color in CIE L*a*b* (Table 1)
TAU_ACM = 20.0               # CIEDE2000 distance threshold (Table 1)


def reference_distance(lab: np.ndarray, roi: np.ndarray, c_ref=C_REF) -> np.ndarray:
    """CIEDE2000 distance of every pixel to C_ref (infinite outside the ROI)."""
    dist = delta_e00(lab, np.asarray(c_ref, np.float32))
    dist[~roi] = np.inf
    return dist


class AdaptiveColorMorphology:
    def __init__(self, lab: np.ndarray, roi: np.ndarray, c_ref=C_REF, tau=TAU_ACM, dist=None):
        self.lab = lab
        self.tau = tau
        self.c_ref = np.asarray(c_ref, np.float32)
        # distance of every pixel to the reference exudate color
        self.dist = reference_distance(lab, roi, c_ref) if dist is None else dist

        # rank of every pixel by distance (0 = closest to C_ref)
        flat = self.dist.ravel()
        self.order = np.argsort(flat, kind="stable")
        rank = np.empty(flat.size, np.int64)
        rank[self.order] = np.arange(flat.size)
        self.rank = rank.reshape(self.dist.shape).astype(np.float32)  # exact below 2**24 px
        self.colors = lab.reshape(-1, 3)

    def _winner(self, rank_img: np.ndarray) -> np.ndarray:
        """Flat index of the pixel holding each rank."""
        return self.order[rank_img.astype(np.int64)]

    def dilate(self, se: np.ndarray) -> np.ndarray:
        """Eq. (17)."""
        q = self._winner(cv2.erode(self.rank, se, borderType=cv2.BORDER_REPLICATE))  # min distance
        apply = self.dist.ravel()[q].reshape(self.dist.shape) < self.tau
        out = self.lab.copy()
        out[apply] = self.colors[q.reshape(self.dist.shape)[apply]]
        return out

    def erode(self, se: np.ndarray) -> np.ndarray:
        """Eq. (18)."""
        q = self._winner(cv2.dilate(self.rank, se, borderType=cv2.BORDER_REPLICATE))  # max distance
        apply = self.dist < self.tau
        out = self.lab.copy()
        out[apply] = self.colors[q.reshape(self.dist.shape)[apply]]
        return out

    def gradient(self, se: np.ndarray) -> np.ndarray:
        """Eq. (19): G = CIEDE2000(A dilate B, A erode B)."""
        return delta_e00(self.dilate(se), self.erode(se))
