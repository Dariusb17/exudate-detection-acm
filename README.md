# Exudate Detection in Color Fundus Images by Adaptive Color Morphology

Python implementation of the exudate-detection part (Section III-A, Algorithm 1) of:

> M. Rostamian, M. Shekari, *Designing an Adaptive Color Morphology Operator for
> Exudates and Microaneurysms Detection in Color Fundus Images*, IEEE Access,
> vol. 14, pp. 54937–54953, 2026. doi:10.1109/ACCESS.2026.3680927

Team: Borda Darius, Jecan Alexandru, Ana Mustatia.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Datasets

| Dataset | Used for | Where |
|---|---|---|
| **DIARETDB1** (89 images, expert exudate maps) | the paper's own benchmark (image-level, Table 3) and the histogram-matching reference image `image006.png` | official zip `diaretdb1_v_1_1.zip` (LUT University; the original site is offline, an archived copy is on the Internet Archive) → extract to `data/DIARETDB1/` |
| **IDRiD** segmentation set (81 images, pixel-level hard-exudate and optic-disc masks, CC-BY 4.0) | pixel-level evaluation; choosing the values the paper does not give (training images only) | `A. Segmentation.zip` from <https://zenodo.org/records/17219542> → extract to `data/IDRiD/` |

The reference color statistics (`exudates/reference_stats.json`) are already
included, so the detector works on any image without the datasets. To regenerate them from DIARETDB1:

```bash
python make_reference.py        # writes exudates/reference_stats.json
```

## Exora: the web app (easiest)

**Exora** is the web interface for the detector. The name comes from *exudate* +
*ora serrata*, the edge of the retina.

Double-click **`start.bat`** (Windows; the first run creates `.venv` and installs
everything), or run:

```bash
python app.py                   # opens http://127.0.0.1:7860
```

Drop a fundus photo on the page or pick a sample. Exora shows:

- the verdict for the whole photo and the marked area,
- a before/after slider with the detected exudates in blue, and a download of the result,
- every stage of Algorithm 1, with a short explanation of each,
- for DIARETDB1 and IDRiD photos with expert annotations in `data/`, a comparison with the experts.

The detection settings (τ_ACM, image rule) can be changed on the page; *Use the paper's τ = 20*
switches to the paper's value. The interface only displays results: all analysis is done by
`exudates/detection.py`, the same code the evaluation scripts use.

| File | What it is |
|---|---|
| [app.py](app.py) | small web server (FastAPI) that runs the detector on an uploaded photo |
| [web/](web/) | the page: `index.html`, `style.css`, `app.js` |
| [examples/](examples/) | sample photos from IDRiD (P. Porwal et al., IEEE Dataport 2018, doi:10.21227/H25W98, CC BY 4.0) |

## Command line

```bash
# one image or a folder (--steps saves a figure with every stage of Algorithm 1)
python detect.py "data/DIARETDB1/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image015.png" --steps

python evaluate_ddb1.py            # image-level SE / SP / accuracy on DIARETDB1 (as in the paper)
python evaluate.py --split test    # pixel-level on the IDRiD test set
python evaluate_od.py              # optic disc localization vs. IDRiD OD masks
python tune.py                     # re-select the unspecified values (IDRiD training set)

# any of the scripts with the paper's exact tau_ACM = 20:
python evaluate_ddb1.py --paper-values
```

## Pipeline (Algorithm 1)

| Stage | Paper | Code |
|---|---|---|
| 1. Pre-processing | RGB → CIE L\*a\*b\*; histogram matching (mean & std of each channel) to the fixed reference DIARETDB1 image006, ROI pixels only | [preprocessing.py](exudates/preprocessing.py) |
| 2. Candidate extraction (ACM) | CIEDE2000 distance to C_ref = (85, 10, 40); adaptive dilation (Eq. 17), adaptive erosion (Eq. 18), gradient G = ΔE00(dilation, erosion) (Eq. 19); G > 0.15; hole filling | [ciede2000.py](exudates/ciede2000.py), [color_morphology.py](exudates/color_morphology.py) |
| 3. Confounding structures | optic disc (entropy-and-shape heuristic), vessels (closing of the green channel, SE = 1.25 × largest vessel width), ROI border; B_clean = B_filled ∧ ¬(OD ∨ vessels ∨ border) | [anatomy.py](exudates/anatomy.py) |
| 4. Segmentation & refinement | mask = green channel, marker = mask − candidates, reconstruction by dilation, (mask − reconstruction) > τ_recon, remove regions with sparse Sobel edge pixels, 5×5 dilation | [detection.py](exudates/detection.py) |

CIEDE2000 (Eqs. 1–16) is implemented directly from the paper's formulas and
checked against the published test values of Sharma et al. (2005).

### What the paper specifies vs. what we had to choose

| Setting | Paper | Our value | Why |
|---|---|---|---|
| Reference image | DIARETDB1 image006 | same | — |
| C_ref | (85, 10, 40) | same | — |
| τ_binarize | 0.15 | same | — |
| Final dilation | 5×5 | same | — |
| **τ_ACM** | **20** | **28** | after color matching, real exudates are 20–35 ΔE00 from C_ref (background ≈ 44), so with 20 only the brightest exudate pixels and the optic disc pass. Run with `--paper-values` for 20. |
| Image "with exudates" rule | not given | ≥ 2000 px marked | chosen with `tune_tau.py` on IDRiD training images |
| G normalization before 0.15 | not given | divide by the image maximum | |
| ACM window | not given | 5×5 | |
| Candidate dilation ("filling candidate regions with dilation", Fig. 1) | not given | 4 px | |
| τ_recon | not given | 10 gray levels | |
| Sobel threshold / min. edge-pixel density | not given | 30 / 0.3 | |
| Optic disc entropy window, threshold | not given; paper keeps **low**-entropy candidates | radius 5.3 % of the ROI diameter, top 5 % entropy | on IDRiD the low-entropy area is the macula (0/81 correct); the OD is the highest-entropy spot (75/81 correct) |
| Vessel width, vessel threshold | not given | largest vessel ≈ 1.2 % of ROI diameter; Otsu | |
| Standard FOV | "resized to a fixed FOV", w_max = D_ROI/99 ≈ 15 px | ROI diameter 1500 px | |

The unspecified values were chosen with `tune.py` on 15 IDRiD **training**
images only (pixel F1); IDRiD test and DIARETDB1 were not used for this.

## Results

**DIARETDB1, image-level** (89 images, 38 with exudates at expert confidence
≥ 0.75; nothing on this dataset was used to choose any value):

| Setting | Sensitivity | Specificity | Accuracy |
|---|---|---|---|
| Paper (Table 3) | 0.97 | 0.91 | 0.93 |
| Ours, τ_ACM = 28, image positive if ≥ 2000 px marked | 0.42 | 0.92 | 0.71 |
| Ours, same τ, ≥ 250 px marked | 0.82 | 0.80 | 0.81 |
| Paper's τ_ACM = 20, ≥ 25 px marked | 0.58 | 0.84 | 0.73 |

The paper does not state when an image counts as "with exudates"; the result
depends strongly on that rule (`evaluate_ddb1.py` prints the full table).
The 2000 px rule was chosen on IDRiD, whose lesions are larger than DIARETDB1's.

**IDRiD test set, pixel-level** (27 images): sensitivity 0.33, PPV 0.79, F1 0.46.

**Optic disc** (entropy method, IDRiD masks): 74/81 (91 %).

τ_ACM and the image rule were chosen with `tune_tau.py` on 15 IDRiD training
images with exudates + 25 healthy IDRiD training images (grade 0).
