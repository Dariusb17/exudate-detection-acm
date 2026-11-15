"""Adaptive color morphology (Eqs. 17-19) on small synthetic images."""
import numpy as np

from exudates.ciede2000 import delta_e00
from exudates.color_morphology import C_REF, TAU_ACM, AdaptiveColorMorphology

BACKGROUND = (40.0, 30.0, 40.0)      # retina-like orange, far from C_ref
SE3 = np.ones((3, 3), np.uint8)


def patch_image(size=15, lo=5, hi=10):
    """Background with a square of exactly the exudate color at [lo:hi, lo:hi]."""
    lab = np.empty((size, size, 3), np.float32)
    lab[:] = BACKGROUND
    lab[lo:hi, lo:hi] = C_REF
    return lab, np.ones((size, size), bool)


def random_image(size, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(30, 95, (size, size)), rng.uniform(0, 40, (size, size)),
                     rng.uniform(10, 60, (size, size))], axis=-1).astype(np.float32)


def brute_force(lab, tau, erode):
    """Eqs. (17) and (18) written as plain loops over a 3x3 window."""
    h, w = lab.shape[:2]
    dist = delta_e00(lab, np.asarray(C_REF, np.float32))
    out = lab.copy()
    for y in range(h):
        for x in range(w):
            ys = slice(max(y - 1, 0), min(y + 2, h))
            xs = slice(max(x - 1, 0), min(x + 2, w))
            win_d, win_c = dist[ys, xs].reshape(-1), lab[ys, xs].reshape(-1, 3)
            if erode:
                if dist[y, x] < tau:
                    out[y, x] = win_c[np.argmax(win_d)]
            elif win_d.min() < tau:
                out[y, x] = win_c[np.argmin(win_d)]
    return out


def test_background_is_far_from_reference():
    assert float(delta_e00(np.array(BACKGROUND), np.array(C_REF))) > TAU_ACM


def test_dilation_grows_the_exudate_patch_by_one_pixel():
    lab, roi = patch_image()
    out = AdaptiveColorMorphology(lab, roi).dilate(SE3)
    is_ref = np.all(out == np.float32(C_REF), axis=-1)
    expected = np.zeros((15, 15), bool)
    expected[4:11, 4:11] = True
    assert np.array_equal(is_ref, expected)


def test_erosion_shrinks_the_patch_and_leaves_the_background_alone():
    lab, roi = patch_image()
    out = AdaptiveColorMorphology(lab, roi).erode(SE3)
    is_ref = np.all(out == np.float32(C_REF), axis=-1)
    expected = np.zeros((15, 15), bool)
    expected[6:9, 6:9] = True
    assert np.array_equal(is_ref, expected)
    assert np.all(out[~expected] == np.float32(BACKGROUND))


def test_gradient_is_high_on_the_patch_border_and_zero_elsewhere():
    lab, roi = patch_image()
    g = AdaptiveColorMorphology(lab, roi).gradient(SE3)
    border = np.zeros((15, 15), bool)
    border[4:11, 4:11] = True
    border[6:9, 6:9] = False
    assert np.all(g[border] > TAU_ACM)
    assert np.all(g[~border] == 0)


def test_no_new_colors_are_created():
    lab = random_image(20, seed=1)
    acm = AdaptiveColorMorphology(lab, np.ones((20, 20), bool), tau=30.0)
    original = {tuple(c) for c in lab.reshape(-1, 3)}
    for out in (acm.dilate(SE3), acm.erode(SE3)):
        assert {tuple(c) for c in out.reshape(-1, 3)} <= original


def test_matches_the_equations_written_as_loops():
    """The fast rank-filter implementation gives the same result as Eqs. (17)-(18) applied directly."""
    lab = random_image(24, seed=2)
    acm = AdaptiveColorMorphology(lab, np.ones((24, 24), bool), tau=30.0)
    assert np.array_equal(acm.dilate(SE3), brute_force(lab, 30.0, erode=False))
    assert np.array_equal(acm.erode(SE3), brute_force(lab, 30.0, erode=True))


def test_pixels_outside_the_roi_never_count_as_exudate():
    lab, roi = patch_image()
    roi[:, :8] = False                       # left half, including part of the patch, is outside
    out = AdaptiveColorMorphology(lab, roi).dilate(SE3)
    assert np.all(out[:, :4] == np.float32(BACKGROUND))
