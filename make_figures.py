"""Figures for the report, written to docs/figures/.

    python make_figures.py

  steps_<photo>.jpg     every stage of Algorithm 1 on a sample photo
  modes_<photo>.jpg     the same photo with Algorithm 1 taken literally, with our
                        first edge filter and with the current settings
  results_ddb1.png      DIARETDB1 image-level results next to the paper's
                        (needs results/eval_ddb1*/summary.json, see evaluate_ddb1.py)

The sample photos are the IDRiD images in examples/ (CC BY 4.0). The expert
masks are used when IDRiD is in data/.
"""
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exudates import Params, detect_exudates, overlay
from exudates.evaluation import load_gt_mask, load_rgb
from exudates.preprocessing import preprocess
from exudates.visualize import step_figure

OUT = Path("docs/figures")
EXAMPLES = Path("examples")
IDRID_GT = Path("data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths")
PAPER_DDB1 = {"sensitivity": 0.97, "specificity": 0.91, "accuracy": 0.93}     # Table 3 of the paper

INK, MUTED, GRID = "#121A21", "#56626D", "#D3DADF"
BAR_COLORS = {"Paper (Table 3)": "#9AA5AE", "Algorithm 1 literally": "#E0A458", "Ours": "#1D5FD1"}


def expert_mask(name, pre):
    """IDRiD hard-exudate mask for this photo, or None if the dataset is not in data/."""
    for f in IDRID_GT.glob(f"*/3. Hard Exudates/{name}_EX.tif"):
        return load_gt_mask(f, pre) & pre.roi
    return None


def coverage(mask, gt):
    """Share of the expert area that is marked, as text for a panel title."""
    if gt is None or not gt.any():
        return f"{int(mask.sum()):,} px marked"
    return f"{100 * (mask & gt).sum() / gt.sum():.0f}% of the expert area"


def steps_figures(names):
    for name in names:
        res = detect_exudates(load_rgb(EXAMPLES / f"{name}.jpg"), Params())
        step_figure(res, OUT / f"steps_{name}.jpg", gt=expert_mask(name, res.pre), title=name)
        print("wrote", OUT / f"steps_{name}.jpg")


def modes_figure(name):
    """One photo, three versions of the method."""
    pre = preprocess(load_rgb(EXAMPLES / f"{name}.jpg"))
    gt = expert_mask(name, pre)
    versions = [
        ("Algorithm 1 taken literally", Params.paper_strict()),
        ("First version: edges counted over the whole region", replace(Params(), edge_mode="area", edge_density=0.3)),
        ("Current: edges counted on the outline", Params()),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    axes[0].imshow(pre.rgb)
    axes[0].set_title("Input", fontsize=12, color=INK)
    for ax, (title, params) in zip(axes[1:], versions):
        mask = detect_exudates(None, params, pre=pre).mask
        ax.imshow(overlay(pre.rgb, mask))
        ax.set_title(f"{title}\n{coverage(mask, gt)}", fontsize=12, color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"modes_{name}.jpg", dpi=80)
    plt.close(fig)
    print("wrote", OUT / f"modes_{name}.jpg")


def results_figure():
    """Grouped bars: sensitivity, specificity and accuracy on DIARETDB1."""
    series = {"Paper (Table 3)": PAPER_DDB1}
    for label, folder in (("Algorithm 1 literally", "results/eval_ddb1_strict"), ("Ours", "results/eval_ddb1")):
        f = Path(folder) / "summary.json"
        if f.exists():
            series[label] = json.loads(f.read_text())
    if len(series) == 1:
        print("no DIARETDB1 results found - run evaluate_ddb1.py first")
        return
    metrics = ["sensitivity", "specificity", "accuracy"]
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (label, values) in enumerate(series.items()):
        xs = [m + (i - (len(series) - 1) / 2) * width for m in range(len(metrics))]
        bars = ax.bar(xs, [values[m] for m in metrics], width * 0.92, label=label, color=BAR_COLORS[label])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(metrics)), [m.capitalize() for m in metrics], fontsize=11, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("DIARETDB1, image level (89 images)", fontsize=13, color=INK, loc="left", pad=28)
    ax.legend(frameon=False, ncols=len(series), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "results_ddb1.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "results_ddb1.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    steps_figures(["IDRiD_10", "IDRiD_25"])
    modes_figure("IDRiD_10")
    results_figure()
