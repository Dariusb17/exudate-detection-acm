"""Local web interface for the exudate detector.

    python app.py            -> opens http://127.0.0.1:7860 in the browser

Upload a fundus image (or click an example) and the page shows the detected
exudates, the decision for the whole image and every stage of Algorithm 1.
If the image is from DIARETDB1, the expert ground truth is compared as well.
"""
import time
from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from matplotlib import colormaps

from evaluate_ddb1 import ROOT as DDB1_ROOT, load_gt
from exudates import Params, detect_exudates, overlay
from exudates.evaluation import load_rgb
from exudates.visualize import compare_image, step_panels

DDB1 = Path(DDB1_ROOT)
EXAMPLE_NAMES = ["image016", "image014", "image019", "image060"]   # 3 with exudates, 1 healthy
EXAMPLES = [str(DDB1 / "ddb1_fundusimages" / f"{n}.png") for n in EXAMPLE_NAMES]
EXAMPLES = [[e] for e in EXAMPLES if Path(e).exists()]
GALLERY_WIDTH = 700

HOW_IT_WORKS = """
**1 · Color normalization.** The image is converted to CIE L\\*a\\*b\\* and its colors are
matched to a fixed reference image (DIARETDB1 image006), so one exudate color fits every photo.

**2 · Candidates by color (Adaptive Color Morphology).** Every pixel gets its CIEDE2000
distance to the exudate color C_ref = (85, 10, 40). The adaptive dilation keeps the most
exudate-like color in each 5×5 window, the erosion the least exudate-like one; their
difference (the gradient G) is high around yellow lesions. G > 0.15 → candidate regions.

**3 · Remove look-alikes.** Optic disc (highest-entropy region), blood vessels (closing of
the green channel) and the border of the field of view are removed from the candidates.

**4 · Confirm & refine.** Morphological reconstruction on the green channel keeps spots that
are clearly brighter than their surroundings; regions without sharp (Sobel) edges are dropped;
a final 5×5 dilation gives the exudate mask.
"""

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#title h1 {margin-bottom: 0;}
#title p {margin-top: 4px; opacity: .75;}
.verdict {border-radius: 12px; padding: 16px 20px; font-size: 15px; line-height: 1.5;}
.verdict b.big {font-size: 22px; display: block; margin-bottom: 4px;}
.verdict.pos {background: #fdecea; border-left: 6px solid #d93025; color: #5f1410;}
.verdict.neg {background: #e6f4ea; border-left: 6px solid #1e8e3e; color: #0d3d1b;}
.verdict.idle {background: rgba(127,127,127,.08); border-left: 6px solid #9aa0a6;}
"""


def _to_rgb(img, cmap):
    """Panel image -> uint8 RGB for display."""
    if cmap is None:
        out = img
    elif img.dtype == bool:
        out = np.dstack([img.astype(np.uint8) * 255] * 3)
    else:
        x = img.astype(np.float32)
        x = np.clip(x / max(1e-6, float(np.percentile(x, 99.8))), 0, 1)
        out = (colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
    h, w = out.shape[:2]
    if w > GALLERY_WIDTH:
        out = cv2.resize(out, (GALLERY_WIDTH, int(h * GALLERY_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return out


def _ground_truth(path, pre):
    """DIARETDB1 expert map for this image, if the file name matches one."""
    if not path:
        return None
    gt_file = DDB1 / "ddb1_groundtruth" / "hardexudates" / Path(path).name
    return load_gt(gt_file, pre) if gt_file.exists() else None


def run(path, tau_acm, min_area):
    if not path:
        raise gr.Error("Upload a fundus image first (or click one of the examples).")
    t0 = time.time()
    params = replace(Params(), tau_acm=float(tau_acm), image_min_area=int(min_area))
    res = detect_exudates(load_rgb(path), params)
    gt = _ground_truth(path, res.pre)
    secs = time.time() - t0

    px = int(res.mask.sum())
    regions = cv2.connectedComponents(res.mask.astype(np.uint8))[0] - 1
    positive = px >= params.image_min_area
    lines = [f"{px:,} exudate pixels in {regions} region(s) · rule: ≥ {params.image_min_area:,} px · {secs:.1f} s"]
    if gt is not None:
        has_gt = bool(gt.any())
        ok = has_gt == positive
        lines.append(f"Expert ground truth (DIARETDB1): <b>{'exudates' if has_gt else 'no exudates'}</b> "
                     f"→ prediction {'✔ correct' if ok else '✘ wrong'}")
    cls, head = ("pos", "Exudates detected") if positive else ("neg", "No exudates found")
    verdict = f'<div class="verdict {cls}"><b class="big">{head}</b>{"<br>".join(lines)}</div>'

    result = (res.pre.rgb, overlay(res.pre.rgb, res.mask))
    compare = compare_image(res.pre.rgb, res.mask, gt) if gt is not None else None
    gallery = [(_to_rgb(img, cmap), name) for name, img, cmap in step_panels(res, gt)]
    return verdict, result, gallery, gr.update(value=compare, visible=compare is not None)


with gr.Blocks(title="Exudate Detection") as demo:
    gr.Markdown(
        "# Exudate Detection in Color Fundus Images\n"
        "Adaptive Color Morphology — Rostamian & Shekari, *IEEE Access* 2026, Section III-A · "
        "Borda Darius, Jecan Alexandru, Ana Mustatia",
        elem_id="title",
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            inp = gr.Image(type="filepath", label="Fundus image", height=340)
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="DIARETDB1 examples (3 with exudates, 1 healthy)")
            with gr.Accordion("Settings", open=False):
                tau = gr.Slider(15, 45, value=Params().tau_acm, step=1, label="τ_ACM (CIEDE2000 threshold)",
                                info="Paper: 20. Our value: 28 (chosen on IDRiD training images).")
                min_area = gr.Number(value=Params().image_min_area, precision=0,
                                     label="Pixels needed to call the image 'with exudates'")
            btn = gr.Button("Detect exudates", variant="primary", size="lg")
            with gr.Accordion("How it works", open=False):
                gr.Markdown(HOW_IT_WORKS)
        with gr.Column(scale=7, min_width=420):
            verdict = gr.HTML('<div class="verdict idle"><b class="big">Waiting for an image</b>'
                              'Upload a photo or pick an example, then press <i>Detect exudates</i>.</div>')
            with gr.Tabs():
                with gr.Tab("Result"):
                    result = gr.ImageSlider(label="Original ⟷ detected exudates (blue)", height=560)
                    compare = gr.Image(label="vs. expert ground truth: green = correct, yellow = false alarm, "
                                             "red = missed", visible=False, height=560)
                with gr.Tab("Step by step (Algorithm 1)"):
                    gallery = gr.Gallery(columns=3, height=640, object_fit="contain", label="Stages")

    btn.click(run, [inp, tau, min_area], [verdict, result, gallery, compare])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS, theme=gr.themes.Soft(primary_hue="blue"))
