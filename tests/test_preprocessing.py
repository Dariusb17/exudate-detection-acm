"""Pre-processing: field of view, color conversion and histogram matching."""
import cv2
import numpy as np
import pytest

from exudates.preprocessing import (WORK_WIDTH, histogram_matching, lab_stats, lab_to_rgb, load_reference,
                                    preprocess, rgb_to_lab, roi_mask)


def fundus_like(h=300, w=400, radius=140, seed=0):
    """Black frame with an orange disc, like a fundus photo."""
    rng = np.random.default_rng(seed)
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (w // 2, h // 2), radius, 1, -1)
    disc = disc.astype(bool)
    rgb = np.zeros((h, w, 3), np.uint8)
    noise = rng.normal(0, 12, (h, w, 3))
    rgb[disc] = np.clip(np.array([190, 95, 40]) + noise[disc], 0, 255).astype(np.uint8)
    return rgb, disc


def test_roi_mask_finds_the_disc():
    rgb, disc = fundus_like()
    roi = roi_mask(rgb)
    overlap = (roi & disc).sum() / (roi | disc).sum()
    assert overlap > 0.97


def test_lab_round_trip():
    rgb, _ = fundus_like()
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_histogram_matching_reaches_the_reference_statistics():
    rgb, disc = fundus_like()
    ref = {"mean": [45.0, 25.0, 35.0], "std": [8.0, 6.0, 7.0]}
    out = histogram_matching(rgb_to_lab(rgb), disc, ref)
    stats = lab_stats(out, disc)
    assert stats["mean"] == pytest.approx(ref["mean"], abs=0.05)
    assert stats["std"] == pytest.approx(ref["std"], abs=0.05)
    assert np.all(out[~disc] == 0)


def test_reference_statistics_are_the_ones_of_image006():
    ref = load_reference()
    assert ref["source"] == "image006.png"
    assert len(ref["mean"]) == 3 and len(ref["std"]) == 3


def test_preprocess_crops_to_the_retina_and_resizes():
    rgb, _ = fundus_like()
    pre = preprocess(rgb)
    assert pre.rgb.shape[1] == WORK_WIDTH
    assert pre.roi.shape == pre.rgb.shape[:2] == pre.lab.shape[:2]
    # the disc touches the left and right edge of the cropped image
    assert pre.roi[:, 0].any() and pre.roi[:, -1].any()
    assert np.all(pre.rgb_norm[~pre.roi] == 0)
