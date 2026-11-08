"""CIEDE2000 color difference, Eqs. (1)-(16) of the paper (vectorized).

All inputs are CIE L*a*b* arrays with the channels on the last axis
(L in [0, 100]). Arrays broadcast, so one reference color can be compared
with a whole image at once.
"""
from __future__ import annotations

import numpy as np


def delta_e00(lab1: np.ndarray, lab2: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> np.ndarray:
    lab1 = np.asarray(lab1, np.float64)
    lab2 = np.asarray(lab2, np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Eq. (1): chroma of each sample, then their mean
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    # Eq. (2): scaling factor g
    C_bar7 = C_bar ** 7
    g = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))
    # Eqs. (3)-(7): modified a', C', h'
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Eqs. (8)-(10): differences
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    # means used by the weighting functions
    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2
    hsum = h1p + h2p
    hp_bar = np.where(np.abs(h1p - h2p) > 180,
                      np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
                      hsum / 2)
    hp_bar = np.where(C1p * C2p == 0, hsum, hp_bar)

    # Eqs. (11)-(14): weighting functions
    SL = 1 + 0.015 * (Lp_bar - 50) ** 2 / np.sqrt(20 + (Lp_bar - 50) ** 2)
    SC = 1 + 0.045 * Cp_bar
    T = (1 - 0.17 * np.cos(np.radians(hp_bar - 30))
         + 0.24 * np.cos(np.radians(2 * hp_bar))
         + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
         - 0.20 * np.cos(np.radians(4 * hp_bar - 63)))
    SH = 1 + 0.015 * Cp_bar * T

    # Eq. (15): rotation term
    Cp_bar7 = Cp_bar ** 7
    RT = (-2 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((hp_bar - 275) / 25) ** 2)))))

    # Eq. (16)
    tL = dLp / (kL * SL)
    tC = dCp / (kC * SC)
    tH = dHp / (kH * SH)
    return np.sqrt(np.maximum(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH, 0)).astype(np.float32)
