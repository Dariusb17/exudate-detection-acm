"""Exora - web app for the exudate detector.

    python app.py            -> opens http://127.0.0.1:7860 in the browser

The page (web/) talks to the endpoints below. The detection itself is
exudates/detection.py (Algorithm 1 of the paper); this file only passes the
photo in and turns the results into pictures for the page.

  GET  /api/samples              example photos
  GET  /api/samples/{id}         the photo itself
  GET  /api/samples/{id}/thumb   small preview
  POST /api/analyze              run the detector on an uploaded photo
"""
import base64
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps

from evaluate_ddb1 import ROOT as DDB1_ROOT, load_gt as load_ddb1_gt
from exudates import Params, detect_exudates, overlay
from exudates.color_morphology import TAU_ACM
from exudates.evaluation import load_gt_mask
from exudates.visualize import compare_image, step_panels

HERE = Path(__file__).parent
WEB = HERE / "web"
DDB1 = HERE / DDB1_ROOT
IDRID_GT = HERE / "data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths"
PORT = 7860
DISPLAY_WIDTH = 1000          # pictures sent to the page are at most this wide

# Example photos: IDRiD ones ship with the repo (CC BY 4.0), DIARETDB1 ones are
# used when the dataset is in data/.
SAMPLES = {
    "idrid-03": (HERE / "examples/IDRiD_03.jpg", "IDRiD 03", "Exudates"),
    "idrid-10": (HERE / "examples/IDRiD_10.jpg", "IDRiD 10", "Exudates"),
    "idrid-25": (HERE / "examples/IDRiD_25.jpg", "IDRiD 25", "Exudates"),
    "idrid-118": (HERE / "examples/IDRiD_118.jpg", "IDRiD 118", "Healthy"),
    "ddb1-016": (DDB1 / "ddb1_fundusimages/image016.png", "DIARETDB1 016", "Exudates"),
    "ddb1-019": (DDB1 / "ddb1_fundusimages/image019.png", "DIARETDB1 019", "Exudates"),
    "ddb1-060": (DDB1 / "ddb1_fundusimages/image060.png", "DIARETDB1 060", "Healthy"),
}
SAMPLES = {k: v for k, v in SAMPLES.items() if v[0].exists()}

# One entry per picture returned by step_panels(): (stage of Algorithm 1, title, explanation)
STAGES = [
    ("Pre-processing", "Input photo",
     "The original fundus photograph, cropped to the round field of view and resized to a standard "
     "width of 1500 px so every image has the same scale."),
    ("Pre-processing", "Color normalization",
     "Converted to <b>CIE L*a*b*</b> (one lightness axis, two color axes, close to how people see color). "
     "The mean and spread of each channel are matched to a fixed reference photo (DIARETDB1 image006), "
     "so one exudate color works for every camera."),
    ("Candidate extraction", "Distance to the exudate color",
     "Every pixel's <b>CIEDE2000</b> distance to the reference exudate color C<sub>ref</sub> = (85, 10, 40). "
     "CIEDE2000 measures how different two colors look to a person. Yellow = close, dark = far."),
    ("Candidate extraction", "Adaptive color gradient",
     "<b>Adaptive color morphology.</b> Dilation takes the most exudate-like color in each 5×5 window, "
     "erosion the least exudate-like one (only colors within τ<sub>ACM</sub> of C<sub>ref</sub> take part). "
     "Their CIEDE2000 difference, the gradient G, lights up around yellow lesions."),
    ("Candidate extraction", "Gradient threshold",
     "G is scaled to 0–1 and thresholded at <b>0.15</b>, the paper's value. "
     "White pixels outline the exudate-colored regions."),
    ("Candidate extraction", "Filled candidates",
     "Holes inside the outlines are filled, so each candidate becomes a solid region."),
    ("Look-alike removal", "Optic disc, vessels, border",
     "Structures that can be mistaken for exudates: the <b>optic disc</b> (magenta, the region with the "
     "highest local entropy), <b>blood vessels</b> (green, from a morphological closing of the green "
     "channel) and the <b>border</b> of the field of view (cyan)."),
    ("Look-alike removal", "Clean candidates",
     "Candidates on the optic disc, vessels or border are removed. The rest are slightly enlarged so "
     "they cover the whole lesion."),
    ("Segmentation & refinement", "Reconstruction difference",
     "<b>Morphological reconstruction</b> on the green channel: candidates are erased and the background "
     "is grown back into them. The difference shows how much brighter each spot is than its surroundings."),
    ("Segmentation & refinement", "Brightness threshold",
     "Only pixels more than 10 gray levels brighter than their reconstructed background are kept, "
     "which drops spots that have the right color but are not bright lesions."),
    ("Segmentation & refinement", "Edge filter",
     "Exudates have <b>sharp borders</b>. Edges come from the <b>Sobel</b> operator; regions with few "
     "edge pixels (smooth reflections, blur) are removed."),
    ("Segmentation & refinement", "Final mask",
     "A final <b>5×5 dilation</b> gives the exudate mask, painted blue. With at least {min_area:,} marked "
     "pixels the photo is classified as <b>with exudates</b>."),
    ("Expert check", "Compared with the experts",
     "Detection against the experts' annotation: <b>green</b> = found, <b>yellow</b> = marked by Exora "
     "only, <b>red</b> = marked by the experts only."),
]

app = FastAPI(title="Exora")
app.mount("/static", StaticFiles(directory=WEB), name="static")
_thumbs = {}


def data_url(rgb, width=DISPLAY_WIDTH, quality=88):
    """RGB picture -> JPEG data URL, shrunk to `width`."""
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def to_picture(img, cmap):
    """Step result -> RGB picture (masks in black/white, maps through a colormap)."""
    if cmap is None:
        return img
    if img.dtype == bool:
        return np.dstack([img.astype(np.uint8) * 255] * 3)
    x = img.astype(np.float32)
    x = np.clip(x / max(1e-6, float(np.percentile(x, 99.8))), 0, 1)
    return (colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)


def ground_truth(filename, pre):
    """Expert exudate mask if the photo is a DIARETDB1 or IDRiD image we have annotations for."""
    name = Path(filename or "").name
    ddb1 = DDB1 / "ddb1_groundtruth/hardexudates" / name
    if name and ddb1.exists():
        return load_ddb1_gt(ddb1, pre), "DIARETDB1 experts"
    for f in IDRID_GT.glob(f"*/3. Hard Exudates/{Path(name).stem}_EX.tif"):
        return load_gt_mask(f, pre) & pre.roi, "IDRiD experts"
    return None, None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config():
    """Default settings, straight from the detector's parameters."""
    return {"tau_acm": Params.tau_acm, "paper_tau_acm": TAU_ACM, "min_area": Params.image_min_area}


@app.get("/api/samples")
def samples():
    return [{"id": k, "label": label, "kind": kind, "file": path.name,
             "url": f"/api/samples/{k}", "thumb": f"/api/samples/{k}/thumb"}
            for k, (path, label, kind) in SAMPLES.items()]


@app.get("/api/samples/{sid}")
def sample(sid: str):
    if sid not in SAMPLES:
        raise HTTPException(404, "Unknown sample")
    return FileResponse(SAMPLES[sid][0], filename=SAMPLES[sid][0].name)


@app.get("/api/samples/{sid}/thumb")
def sample_thumb(sid: str):
    if sid not in SAMPLES:
        raise HTTPException(404, "Unknown sample")
    if sid not in _thumbs:
        bgr = cv2.imdecode(np.fromfile(str(SAMPLES[sid][0]), np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (240, round(h * 240 / w)), interpolation=cv2.INTER_AREA)
        _thumbs[sid] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return Response(_thumbs[sid], media_type="image/jpeg")


@app.post("/api/analyze")
def analyze(image: UploadFile = File(...), tau_acm: float = Form(Params.tau_acm),
            min_area: int = Form(Params.image_min_area)):
    bgr = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "This file can't be read as an image. Use a JPG, PNG or TIF photo.")
    params = replace(Params(), tau_acm=float(np.clip(tau_acm, 10, 45)), image_min_area=max(0, int(min_area)))

    t0 = time.time()
    try:
        res = detect_exudates(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), params)   # the paper's algorithm
    except Exception:
        raise HTTPException(422, "No retina was found in this photo. Use a color fundus photograph "
                                 "with the round field of view visible.")
    seconds = time.time() - t0

    pre, mask = res.pre, res.mask
    gt, gt_source = ground_truth(image.filename, pre)
    marked = int(mask.sum())
    positive = marked >= params.image_min_area

    stages = []
    for i, (name, img, cmap) in enumerate(step_panels(res, gt)):
        group, title, text = STAGES[i]
        stages.append({"group": group, "title": title, "text": text.format(min_area=params.image_min_area),
                       "image": data_url(to_picture(img, cmap))})

    expert = None
    if gt is not None:
        has = bool(gt.any())
        tp = int((mask & gt).sum())
        expert = {"source": gt_source, "has_exudates": has, "agrees": has == positive,
                  "found_pct": round(100 * tp / max(int(gt.sum()), 1), 1),
                  "correct_pct": round(100 * tp / max(marked, 1), 1),
                  "image": data_url(compare_image(pre.rgb, mask, gt))}

    return {
        "positive": positive,
        "stats": {"marked_px": marked,
                  "regions": cv2.connectedComponents(mask.astype(np.uint8))[0] - 1,
                  "retina_pct": round(100 * marked / max(int(pre.roi.sum()), 1), 2),
                  "min_area": params.image_min_area, "tau_acm": params.tau_acm,
                  "seconds": round(seconds, 1)},
        "original": data_url(pre.rgb),
        "result": data_url(overlay(pre.rgb, mask)),
        "result_full": data_url(overlay(pre.rgb, mask), width=pre.rgb.shape[1], quality=92),
        "stages": stages,
        "expert": expert,
    }


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"Exora is running at {url}  (close this window to stop it)")
    threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
