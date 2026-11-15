"""Detection pipeline: the edge filter on synthetic masks, and Algorithm 1 on a sample photo."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from exudates import Params, detect_exudates
from exudates.detection import _keep_edge_dense_regions
from exudates.evaluation import load_rgb

SAMPLE = Path(__file__).parent.parent / "examples" / "IDRiD_25.jpg"     # a photo with many exudates


def square_with_sharp_outline(size=100, lo=20, hi=80):
    """A large solid region whose strong edges lie only on its outline."""
    mask = np.zeros((size, size), bool)
    mask[lo:hi, lo:hi] = True
    edges = np.zeros((size, size), bool)
    edges[lo - 1:hi + 1, lo - 1:hi + 1] = True
    edges[lo + 2:hi - 2, lo + 2:hi - 2] = False
    return mask, edges


def test_outline_mode_keeps_a_large_region_with_a_sharp_outline():
    mask, edges = square_with_sharp_outline()
    assert np.array_equal(_keep_edge_dense_regions(mask, edges, 0.3, "outline"), mask)


def test_area_mode_removes_the_same_region():
    mask, edges = square_with_sharp_outline()
    assert not _keep_edge_dense_regions(mask, edges, 0.3, "area").any()


@pytest.mark.parametrize("mode", ["outline", "area"])
def test_regions_without_edges_are_removed(mode):
    mask = np.zeros((60, 60), bool)
    mask[10:20, 10:20] = True               # no edges here
    mask[40:46, 40:46] = True               # edges everywhere here
    edges = np.zeros((60, 60), bool)
    edges[38:48, 38:48] = True
    kept = _keep_edge_dense_regions(mask, edges, 0.3, mode)
    assert not kept[10:20, 10:20].any()
    assert kept[40:46, 40:46].all()


@pytest.fixture(scope="module")
def result():
    return detect_exudates(load_rgb(SAMPLE), Params())


def test_mask_is_a_boolean_image_inside_the_retina(result):
    assert result.mask.dtype == bool
    assert result.mask.shape == result.pre.roi.shape
    assert not (result.mask & ~result.pre.roi).any()


def test_candidates_avoid_the_optic_disc_and_the_border(result):
    s = result.steps
    assert s["od"].any(), "the optic disc should be found on this photo"
    assert not (s["b_clean"] & (s["od"] | s["border"])).any()


def test_each_stage_only_removes_pixels(result):
    s = result.steps
    assert not (s["e_init"] & ~s["b_clean"]).any()
    assert not (s["e_refined"] & ~s["e_init"]).any()


def test_photo_with_exudates_is_called_positive(result):
    assert result.mask.sum() >= Params().image_min_area


def test_pixel_mode_keeps_only_edge_pixels(result):
    """edge_mode="pixel" is Algorithm 1 line 21 taken literally: E_init AND E_edges."""
    p = replace(Params(), edge_mode="pixel")
    res = detect_exudates(None, p, pre=result.pre)
    expected = res.steps["e_init"] & (res.steps["sobel"] > p.tau_edge)
    assert np.array_equal(res.steps["e_refined"], expected)
