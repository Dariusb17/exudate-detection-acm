"""Choose the parameters the paper leaves unspecified, on the IDRiD *training* images.

    python tune.py --n-images 20 --trials 60

The test images are never used here, so the test-set score in evaluate.py is
an honest estimate. Prints the best parameters found (by pixel F1).
"""
import argparse
import random
from dataclasses import replace
from pathlib import Path

from exudates import Params, detect_exudates
from exudates.evaluation import lesion_metrics, load_gt_mask, load_rgb, pixel_metrics, summarize
from exudates.preprocessing import preprocess
from evaluate import SPLITS, find_root

# Only the settings the paper does NOT specify are searched. The paper's own
# values (C_ref, tau_ACM, tau_binarize, 5x5 final dilation) stay fixed.
SPACE = dict(
    tau_acm=[20.0, 35.0, 40.0, 45.0, 50.0, 60.0],   # paper: 20 (also searched, see README)
    g_norm="max 20 30 40 60".split(),
    acm_window=[3, 5],
    cand_dilate=[2, 4, 6, 8],
    tau_recon=[4.0, 6.0, 8.0, 10.0, 14.0, 18.0],
    tau_edge=[15.0, 20.0, 30.0, 40.0, 60.0],
    edge_density=[0.0, 0.1, 0.2, 0.3, 0.5],
)


def score(params, data):
    rows = []
    for name, pre, gt in data:
        res = detect_exudates(None, params, pre=pre)
        rows.append(dict(image=name, **pixel_metrics(res.mask, gt, pre.roi), **lesion_metrics(res.mask, gt)))
    return summarize(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/IDRiD")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-paper", action="store_true", help="do not search tau_ACM / G normalization")
    args = ap.parse_args()

    root = find_root(Path(args.data))
    img_dir = root / "1. Original Images" / SPLITS["train"]
    gt_dir = root / "2. All Segmentation Groundtruths" / SPLITS["train"] / "3. Hard Exudates"
    files = sorted(img_dir.glob("*.jpg"))
    files = files[:: max(1, len(files) // args.n_images)][: args.n_images]

    print(f"Preprocessing {len(files)} training images...")
    data = []
    for f in files:
        pre = preprocess(load_rgb(f))
        data.append((f.stem, pre, load_gt_mask(gt_dir / f"{f.stem}_EX.tif", pre) & pre.roi))

    rng = random.Random(args.seed)
    best_p = Params()
    best = score(best_p, data)
    print(f"default: F1={best['pixel_f1']:.3f}")
    if args.fixed_paper:   # keep the paper's tau_ACM and normalization
        for k in ("tau_acm", "g_norm"):
            SPACE.pop(k)
    for t in range(args.trials):
        # 3 of 4 trials change one parameter of the current best, 1 of 4 samples freely
        if t % 4:
            k = rng.choice(list(SPACE))
            cand = replace(best_p, **{k: rng.choice(SPACE[k])})
        else:
            cand = Params(**{k: rng.choice(v) for k, v in SPACE.items()})
        s = score(cand, data)
        if s["pixel_f1"] > best["pixel_f1"]:
            best_p, best = cand, s
            print(f"trial {t:3d}: F1={s['pixel_f1']:.3f} SE={s['pixel_sensitivity']:.3f} "
                  f"PPV={s['pixel_ppv']:.3f} lesionSE={s['lesion_sensitivity']:.3f}  "
                  + ", ".join(f"{k}={getattr(cand, k)}" for k in SPACE))

    print("\nBest parameters on the training set:")
    for k in SPACE:
        print(f"  {k} = {getattr(best_p, k)}")
    print({k: round(v, 3) for k, v in best.items()})


if __name__ == "__main__":
    main()
