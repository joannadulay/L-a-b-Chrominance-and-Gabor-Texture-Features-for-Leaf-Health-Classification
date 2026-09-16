"""
ml/predict_runner_stageD.py

CLI prediction runner for the Stage D (lean) GBM model — baseline +
L*a*b* chrominance + Gabor filter bank (feature_extractor_stageD_lean.py).

Called by ui/ui_main_stageD.py as a subprocess:

    python predict_runner_stageD.py /path/to/captured_image.jpg

Always prints exactly one JSON object on the LAST line of stdout (the UI
parses stdout.splitlines()[-1]) and exits 0, even on a handled failure
(bad image, no leaf, low confidence, missing model, etc.) so the UI can
show a specific reason instead of a generic subprocess error. Non-zero
exit + stderr is reserved for truly unexpected crashes.

What changed in this version
----------------------------
1. Visual metrics. The runner now measures what the leaf actually looks
   like (yellow / green / brown / dark area, spot count, spot shape, edge
   density, mottling) from the segmented leaf pixels and ships those
   numbers inside payload["details"]. The "Visual Findings" tab reads
   exactly these keys, which is why it previously had nothing to show and
   fell back to a placeholder line.
2. Named timings. payload["timings"] now uses the display names the
   details window expects ("Image Loading", "Segmentation", ...), plus a
   "Total Pipeline" entry and the raw *_ms keys for anything still
   reading those.
3. Segmentation output. Alongside the combined figure, the binary mask
   and the masked-out leaf are written as their own PNGs so the details
   window can show the segmentation step on its own tab.

Expected on-disk layout (relative to this file, ml/):
    ml/
      predict_runner_stageD.py   <- this file
      feature_extractor_stageD_lean.py
      feature_extractor.py       <- Article 1 baseline, for count_leaflike_components
      models/
        gbm_model_cv_stageD_lean.pkl
        gate_model.pkl           <- optional, reused from Article 1
      outputs/
        last_analysis.png        <- combined figure, shown by MORE DETAILS
        last_original.png        <- captured leaf, resized
        last_mask.png            <- binary segmentation mask
        last_segmented.png       <- leaf pixels only, background removed
"""

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

# Make sibling modules importable regardless of the caller's cwd (ui_main
# runs this with cwd=BASE_DIR, not cwd=ml/).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor_stageD_lean import extract_features_from_bgr, segment_leaf

try:
    # Scene-level "how many leaf-like blobs" check is unchanged from
    # Article 1 — reuse it if the baseline module is present alongside.
    from feature_extractor import count_leaflike_components
except Exception as _exc:  # pragma: no cover - fallback for standalone use
    print(
        f"[predict_runner_stageD] feature_extractor.py (baseline) not found "
        f"or failed to import ({_exc}); scene-level leaf-count check disabled.",
        file=sys.stderr,
    )

    def count_leaflike_components(image_bgr):
        return 1


MODEL_PATH = SCRIPT_DIR / "models" / "gbm_model_cv_stageD_lean.pkl"
GATE_MODEL_PATH = SCRIPT_DIR / "models" / "gate_model.pkl"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
VIS_PATH = OUTPUT_DIR / "last_analysis.png"
ORIGINAL_PATH = OUTPUT_DIR / "last_original.png"
MASK_PATH = OUTPUT_DIR / "last_mask.png"
SEGMENTED_PATH = OUTPUT_DIR / "last_segmented.png"

IMG_SIZE = 128
# The metrics/visualisation copy of the frame is kept larger than the
# model's 128px input: spot count and circularity are much steadier at
# this size, and the figure stops looking pixelated on screen.
VIS_SIZE = 384

MIN_LEAF_RATIO = 0.03
MAX_LEAF_RATIO = 0.65
CONFIDENCE_THRESHOLD = 0.5
GATE_THRESHOLD = 0.5

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "discolored": "Discolored",
    "diseased": "Diseased",
}

NO_LEAF_LABEL = "No Kangkong Detected"
UNCLEAR_LABEL = "Unclear Leaf Image"

# --- Capture quality gate --------------------------------------------------
# A model will happily return 93% confidence on a blurry, half-framed clump
# of foliage, because nothing in the feature vector says "this photo is no
# good". These checks run after segmentation and before classification, and
# reject the frame outright rather than letting a confident-looking wrong
# answer through.
#
# All of them are measured on the VIS_SIZE (384px) copy, so the numbers are
# comparable between runs. Tune them against your own captures: print the
# "quality" block from the JSON payload for a few good and bad photos and
# move the threshold to sit between the two groups.

# Variance of the Laplacian over leaf pixels. Sharp leaf detail scores in
# the hundreds; a motion-blurred or out-of-focus frame collapses toward 0.
# This depends heavily on your specific camera, lens focus distance and
# lighting, so 30 is a starting point, not a rule -- tune it against your
# own hardware:
#   1. Capture a handful of leaves you'd consider "good enough" and a
#      handful you'd consider "too blurry".
#   2. Open Details -> Findings on each and read the "Sharpness ..." line.
#   3. Set MIN_SHARPNESS a little below the lowest good score (and above
#      the highest bad one). If the two groups overlap, the camera itself
#      needs a fixed focal distance or a cleaner lens more than the number
#      needs adjusting.
MIN_SHARPNESS = 30.0

# Share of the frame's border band covered by the leaf mask. A leaf sitting
# inside the frame barely touches it; a leaf running off the edge (or a
# scene that is all foliage) covers a lot of it.
MAX_BORDER_CONTACT = 0.25

# Largest mask blob / convex hull of that blob. A single leaf is a smooth,
# compact shape. Overlapping stems and background foliage produce a ragged
# mask with a much smaller solidity.
MIN_SOLIDITY = 0.70

# Largest blob / total mask area. Below this, the mask is really several
# separate things rather than one leaf.
MIN_LARGEST_REGION_SHARE = 0.65

# Exposure limits on leaf pixels: too dark to read, blown out by glare.
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 235.0
MAX_OVEREXPOSED = 0.25
MAX_UNDEREXPOSED = 0.35

MODEL_DISPLAY_NAME = "Stage D GBM — baseline + L*a*b* + Gabor"

# Figure styling, kept in step with the UI theme so the panel shown inside
# the app doesn't look like it came from a different program.
FIG_BG = "#0D1411"
FIG_PANEL = "#1B2A23"
FIG_TEXT = "#E9F2EC"
FIG_MUTED = "#7F9689"
FIG_GRID = "#2C4137"
CLASS_COLORS = {
    "Healthy": "#57C273",
    "Discolored": "#E3A72F",
    "Diseased": "#D1495B",
    "Unclear Leaf Image": "#E07A3F",
    "No Kangkong Detected": "#D1495B",
}


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_models(timings: dict):
    t0 = _now_ms()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Stage D model not found: {MODEL_PATH}")
    model_data = joblib.load(MODEL_PATH)
    gate_data = joblib.load(GATE_MODEL_PATH) if GATE_MODEL_PATH.exists() else None
    timings["Model Loading"] = round(_now_ms() - t0, 2)
    timings["model_load_ms"] = timings["Model Loading"]
    return model_data, gate_data


# ---------------------------------------------------------------------------
# Visual metrics — what the leaf actually looks like
# ---------------------------------------------------------------------------

def describe_leaf_appearance(image_bgr, mask):
    """Measure human-describable properties of the segmented leaf.

    Everything here is computed over leaf pixels only (mask > 0), so the
    background can't inflate a ratio. Key names match what the details
    window looks for, so adding a metric here is enough to make it appear
    in the Visual Findings tab.

    These are colour and shape measurements, not a detector for holes or
    punctures in the tissue: brown and dark pixels are reported as
    discoloration and necrosis, nothing more.
    """
    metrics = {}
    if mask is None or image_bgr is None:
        return metrics

    mask_bool = np.asarray(mask) > 0
    leaf_px = int(np.count_nonzero(mask_bool))
    if leaf_px < 64:
        return metrics

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)      # OpenCV hue is 0-179
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    green = mask_bool & (hue >= 35) & (hue <= 85) & (sat >= 55) & (val >= 40)
    yellow = mask_bool & (hue >= 20) & (hue < 35) & (sat >= 70) & (val >= 90)
    brown = mask_bool & (((hue >= 5) & (hue < 20)) | (hue >= 172)) & (sat >= 45) & (val < 175)
    dark = mask_bool & (val < 60)

    metrics["green_ratio"] = round(float(np.count_nonzero(green)) / leaf_px, 4)
    metrics["yellow_ratio"] = round(float(np.count_nonzero(yellow)) / leaf_px, 4)
    metrics["dark_ratio"] = round(float(np.count_nonzero(dark)) / leaf_px, 4)

    # A lesion is brown discoloration or near-black necrotic tissue. Open
    # the mask first so stray pixels (JPEG noise, the edge of a specular
    # highlight) don't get counted as spots.
    lesion = (brown | dark).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel, iterations=1)
    metrics["lesion_ratio"] = round(float(np.count_nonzero(lesion)) / leaf_px, 4)

    # Spot count / shape: only blobs above 0.2% of the leaf area count, so
    # surface texture isn't reported as a lesion.
    min_area = max(8.0, 0.002 * leaf_px)
    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, circularities = [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        areas.append(area)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter > 0:
            circularities.append(min(1.0, 4.0 * np.pi * area / (perimeter * perimeter)))

    metrics["lesion_count"] = len(areas)
    if areas:
        area_ratios = [a / leaf_px for a in areas]
        metrics["mean_area_ratio"] = round(float(np.mean(area_ratios)), 5)
        metrics["std_area_ratio"] = round(float(np.std(area_ratios)), 5)
        metrics["largest_lesion_ratio"] = round(float(np.max(area_ratios)), 5)
    if circularities:
        metrics["mean_circularity"] = round(float(np.mean(circularities)), 4)

    # Fragmentation: spots per 1% of affected leaf area. Many small spots
    # scattered over a little total area scores high (disease-like); one
    # broad patch scores near zero (stress/chlorosis-like).
    affected_pct = metrics["lesion_ratio"] * 100.0
    metrics["fragmentation"] = (
        round(float(len(areas)) / affected_pct, 3) if affected_pct > 0.01 else 0.0
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    metrics["edge_density"] = round(float(np.count_nonzero(edges[mask_bool])) / leaf_px, 4)

    leaf_gray = gray[mask_bool].astype(np.float32)
    metrics["mean_h"] = round(float(np.mean(hue[mask_bool])), 2)
    metrics["std_h"] = round(float(np.std(hue[mask_bool])), 2)
    metrics["mean_s"] = round(float(np.mean(sat[mask_bool])), 2)
    metrics["mean_v"] = round(float(np.mean(val[mask_bool])), 2)
    metrics["std_gray"] = round(float(np.std(leaf_gray)), 2)

    return metrics


def assess_image_quality(image_bgr, mask):
    """Measure whether this capture is good enough to classify.

    Returns a dict of raw numbers only — the pass/fail decision lives in
    quality_rejection_reason() so the thresholds stay in one place and the
    measurements can be shown in the UI either way.
    """
    quality = {}
    if image_bgr is None or mask is None:
        return quality

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    quality["frame_sharpness"] = round(float(laplacian.var()), 2)

    mask_bool = np.asarray(mask) > 0
    leaf_px = int(np.count_nonzero(mask_bool))
    if leaf_px < 64:
        return quality

    quality["sharpness"] = round(float(laplacian[mask_bool].var()), 2)

    # Border contact: how much of a thin band around the frame is leaf.
    height, width = mask_bool.shape[:2]
    band = max(2, int(0.02 * min(height, width)))
    border = np.zeros_like(mask_bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_px = int(np.count_nonzero(border))
    if border_px:
        quality["border_contact"] = round(
            float(np.count_nonzero(mask_bool & border)) / border_px, 4)

    # Shape of the mask: is it one compact leaf, or a tangle?
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(c)) for c in contours]
    if areas and max(areas) > 0:
        total = float(sum(areas)) or 1.0
        largest_index = int(np.argmax(areas))
        quality["largest_region_share"] = round(areas[largest_index] / total, 4)
        quality["region_count"] = int(sum(1 for a in areas if a >= 0.02 * max(areas)))
        hull_area = float(cv2.contourArea(cv2.convexHull(contours[largest_index])))
        quality["solidity"] = round(areas[largest_index] / hull_area, 4) if hull_area > 0 else 0.0

    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    leaf_value = value[mask_bool]
    quality["mean_brightness"] = round(float(np.mean(leaf_value)), 2)
    quality["overexposed_ratio"] = round(float(np.count_nonzero(leaf_value > 245)) / leaf_px, 4)
    quality["underexposed_ratio"] = round(float(np.count_nonzero(leaf_value < 30)) / leaf_px, 4)

    return quality


def quality_rejection_reason(quality):
    """First reason this capture isn't good enough, or None if it passes.

    Each message names what is wrong and what to do about it, because the
    UI shows this text verbatim and "rejected" on its own doesn't tell
    anyone how to get a usable photo.
    """
    if not quality:
        return None

    sharpness = quality.get("sharpness", quality.get("frame_sharpness"))
    if sharpness is not None and sharpness < MIN_SHARPNESS:
        return (
            f"Image is too blurry to analyze (sharpness {sharpness:.0f}, needs "
            f"{MIN_SHARPNESS:.0f}). Hold the camera still, move slightly further back, "
            "and let it focus before capturing."
        )

    border_contact = quality.get("border_contact")
    if border_contact is not None and border_contact > MAX_BORDER_CONTACT:
        return (
            f"The leaf runs off the edge of the frame ({border_contact:.0%} of the border "
            "is covered). Move the camera back so one whole leaf fits inside the frame."
        )

    largest_share = quality.get("largest_region_share")
    if largest_share is not None and largest_share < MIN_LARGEST_REGION_SHARE:
        return (
            f"Several separate leaf regions are in view (largest is only {largest_share:.0%} "
            "of the detected area). Isolate one leaf against a plain background."
        )

    solidity = quality.get("solidity")
    if solidity is not None and solidity < MIN_SOLIDITY:
        return (
            f"The detected shape is too ragged to be a single leaf (solidity {solidity:.2f}, "
            f"needs {MIN_SOLIDITY:.2f}). Overlapping stems or background foliage are being "
            "picked up — isolate one leaf."
        )

    overexposed = quality.get("overexposed_ratio")
    if overexposed is not None and overexposed > MAX_OVEREXPOSED:
        return (
            f"{overexposed:.0%} of the leaf is blown out by glare. Move the light source or "
            "the leaf so the surface isn't reflecting straight into the lens."
        )

    underexposed = quality.get("underexposed_ratio")
    if underexposed is not None and underexposed > MAX_UNDEREXPOSED:
        return (
            f"{underexposed:.0%} of the leaf is in deep shadow. Add light or move the leaf "
            "out of the shadow before capturing."
        )

    brightness = quality.get("mean_brightness")
    if brightness is not None and brightness < MIN_BRIGHTNESS:
        return (
            f"The leaf is too dark to analyze (brightness {brightness:.0f}). Turn on the grow "
            "light or move to a brighter spot."
        )
    if brightness is not None and brightness > MAX_BRIGHTNESS:
        return (
            f"The leaf is washed out (brightness {brightness:.0f}). Reduce the light or the "
            "camera exposure and capture again."
        )

    return None


def _blank_result():
    return {
        "label": "ML Error",
        "reason": "Image could not be read.",
        "probabilities": {},
        "leaf_ratio": None,
        "component_count": None,
        "confidence": None,
        "gate_leaf_probability": None,
        "visual_metrics": {},
        "quality": {},
        "mask": np.zeros((VIS_SIZE, VIS_SIZE), dtype=np.uint8),
        "image_bgr": np.zeros((VIS_SIZE, VIS_SIZE, 3), dtype=np.uint8),
        "segmented_bgr": np.zeros((VIS_SIZE, VIS_SIZE, 3), dtype=np.uint8),
    }


def run_inference(image_path, model_data, gate_data, timings: dict):
    model = model_data["model"]
    class_names = model_data["class_names"]
    gate_model = gate_data["model"] if gate_data else None

    t0 = _now_ms()
    image_bgr = cv2.imread(str(image_path))
    timings["Image Loading"] = round(_now_ms() - t0, 2)
    timings["image_read_ms"] = timings["Image Loading"]

    if image_bgr is None:
        return _blank_result()

    # Model input stays at IMG_SIZE (that's what the model was trained on);
    # the display/metrics copy is kept larger.
    model_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    vis_bgr = cv2.resize(image_bgr, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)

    t0 = _now_ms()
    scene_component_count = count_leaflike_components(model_bgr)
    timings["Scene Scan"] = round(_now_ms() - t0, 2)
    timings["component_count_ms"] = timings["Scene Scan"]

    t0 = _now_ms()
    mask, leaf_ratio, seg_rejected = segment_leaf(model_bgr)
    timings["Segmentation"] = round(_now_ms() - t0, 2)
    timings["segmentation_ms"] = timings["Segmentation"]

    # Upscale the mask for display/metrics. NEAREST keeps it strictly
    # binary instead of introducing grey edge pixels.
    mask_u8 = np.asarray(mask)
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    elif mask_u8.size and mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255
    vis_mask = cv2.resize(mask_u8, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_NEAREST)
    segmented_bgr = cv2.bitwise_and(vis_bgr, vis_bgr, mask=vis_mask)

    label = NO_LEAF_LABEL
    reason = ""
    probs_dict = {}
    confidence = None
    gate_leaf_prob = None

    # Visual metrics are computed whenever there's a usable mask, not only
    # on an accepted prediction — a rejected frame is exactly when the user
    # wants to know what the image looked like.
    t0 = _now_ms()
    visual_metrics = describe_leaf_appearance(vis_bgr, vis_mask)
    timings["Visual Metrics"] = round(_now_ms() - t0, 2)
    timings["visual_metrics_ms"] = timings["Visual Metrics"]

    t0 = _now_ms()
    quality = assess_image_quality(vis_bgr, vis_mask)
    timings["Quality Check"] = round(_now_ms() - t0, 2)
    timings["quality_check_ms"] = timings["Quality Check"]

    if scene_component_count == 0:
        reason = "No leaf-like region detected."
    elif scene_component_count > 1:
        reason = (
            f"Multiple leaf-like regions detected ({scene_component_count}). "
            "Please isolate one leaf."
        )
    elif seg_rejected:
        reason = f"Leaf region rejected by segmentation (leaf ratio={leaf_ratio:.4f})."
    elif leaf_ratio < MIN_LEAF_RATIO:
        reason = f"Leaf area too small ({leaf_ratio:.4f})."
    elif leaf_ratio > MAX_LEAF_RATIO:
        reason = (
            f"Scene too cluttered or mask too large ({leaf_ratio:.4f}). "
            "Please isolate one leaf."
        )
    elif quality_rejection_reason(quality):
        # Quality gate. Everything below this point assumes a clear, whole,
        # well-lit leaf; without the check a blurry half-leaf still produces
        # a confident class, which is worse than no answer at all.
        label = UNCLEAR_LABEL
        reason = quality_rejection_reason(quality)
    else:
        t0 = _now_ms()
        features = extract_features_from_bgr(model_bgr).reshape(1, -1)
        timings["Feature Extraction"] = round(_now_ms() - t0, 2)
        timings["feature_extraction_ms"] = timings["Feature Extraction"]

        gate_ok = True
        if gate_model is not None:
            t0 = _now_ms()
            gate_leaf_prob = float(gate_model.predict_proba(features)[0][1])
            timings["Gate Model"] = round(_now_ms() - t0, 2)
            timings["gate_inference_ms"] = timings["Gate Model"]
            gate_ok = gate_leaf_prob >= GATE_THRESHOLD
            if not gate_ok:
                reason = f"Rejected by gate model (leaf probability={gate_leaf_prob:.4f})."

        if gate_ok:
            t0 = _now_ms()
            pred_idx = model.predict(features)[0]
            timings["GBM Predict"] = round(_now_ms() - t0, 2)
            timings["predict_ms"] = timings["GBM Predict"]
            pred_label = class_names[pred_idx]

            if hasattr(model, "predict_proba"):
                t0 = _now_ms()
                probs = model.predict_proba(features)[0]
                timings["Predict Probability"] = round(_now_ms() - t0, 2)
                timings["predict_proba_ms"] = timings["Predict Probability"]
                timings["classification_ms"] = round(
                    timings["GBM Predict"] + timings["Predict Probability"], 2
                )

                max_prob = float(np.max(probs))
                probs_dict = {
                    DISPLAY_NAMES.get(cls, cls): round(float(p), 4)
                    for cls, p in zip(class_names, probs)
                }
                if max_prob < CONFIDENCE_THRESHOLD:
                    reason = f"Low model confidence ({max_prob:.4f})."
                else:
                    label = DISPLAY_NAMES.get(pred_label, pred_label)
                    confidence = round(max_prob, 4)
                    reason = f"Accepted prediction (confidence={max_prob:.4f})."
            else:
                timings["classification_ms"] = timings["GBM Predict"]
                label = DISPLAY_NAMES.get(pred_label, pred_label)
                reason = "Prediction made without probability output."

    return {
        "label": label,
        "reason": reason,
        "probabilities": probs_dict,
        "leaf_ratio": round(float(leaf_ratio), 4) if leaf_ratio is not None else None,
        "component_count": scene_component_count,
        "confidence": confidence,
        "gate_leaf_probability": round(gate_leaf_prob, 4) if gate_leaf_prob is not None else None,
        "visual_metrics": visual_metrics,
        "quality": quality,
        "mask": vis_mask,
        "image_bgr": vis_bgr,
        "segmented_bgr": segmented_bgr,
    }


def save_segmentation_images(result) -> dict:
    """Write the three segmentation stages as individual PNGs.

    The details window shows these side by side on its Segmentation tab,
    so each one has to stand alone rather than only existing as a panel
    inside the combined figure.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    writes = [
        ("original_image_path", ORIGINAL_PATH, result.get("image_bgr")),
        ("mask_image_path", MASK_PATH, result.get("mask")),
        ("segmented_image_path", SEGMENTED_PATH, result.get("segmented_bgr")),
    ]
    for key, path, image in writes:
        try:
            if image is not None and cv2.imwrite(str(path), image):
                paths[key] = str(path)
        except Exception as exc:  # pragma: no cover - disk issues only
            print(f"[predict_runner_stageD] could not write {path.name}: {exc}", file=sys.stderr)

    return paths


def save_visualization(result, out_path: Path, timings: dict = None) -> None:
    """Combined analysis figure shown by the UI's MORE DETAILS button.

    Layout:

        [ Original Image ] [ Detected Leaf Mask ] [ Segmented Leaf Region ]
        [ -------------- Prediction Confidence Scores ------------------ ]

    The three image panels are always drawn whenever a mask was produced —
    including for a rejected or low-confidence result, since that's when
    seeing the mask is most useful. Failure here is non-fatal: the UI falls
    back to its own generated summary image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    image_rgb = cv2.cvtColor(result["image_bgr"], cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(result["segmented_bgr"], cv2.COLOR_BGR2RGB)
    mask = result["mask"]

    fig = plt.figure(figsize=(9.6, 6.4), facecolor=FIG_BG)
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.12)

    def image_panel(position, image, title, subtitle="", cmap=None):
        ax = fig.add_subplot(position)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, color=FIG_TEXT, fontsize=11, pad=8)
        ax.set_facecolor(FIG_PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(FIG_GRID)
        if subtitle:
            ax.set_xlabel(subtitle, color=FIG_MUTED, fontsize=8.5, labelpad=6)
        return ax

    ratio = result.get("leaf_ratio")
    ratio_text = f"{ratio:.4f}" if isinstance(ratio, (int, float)) else "--"
    components = result.get("component_count")
    components_text = components if components is not None else "--"

    image_panel(grid[0, 0], image_rgb, "Original Image")
    image_panel(
        grid[0, 1], mask, "Detected Leaf Mask",
        subtitle=f"leaf ratio {ratio_text}   ·   regions {components_text}", cmap="gray",
    )
    image_panel(grid[0, 2], segmented_rgb, "Segmented Leaf Region")

    # --- confidence bars -------------------------------------------------
    ax = fig.add_subplot(grid[1, :])
    ax.set_facecolor(FIG_PANEL)
    probs = result.get("probabilities") or {}

    if probs:
        names = list(probs.keys())
        values = [float(probs[n]) for n in names]
        bar_colors = [CLASS_COLORS.get(n, FIG_MUTED) for n in names]
        bars = ax.bar(names, values, color=bar_colors, width=0.45)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color=FIG_GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}",
                ha="center", color=FIG_TEXT, fontsize=9.5, fontweight="bold",
            )
    else:
        ax.text(
            0.5, 0.5, "No prediction was made for this frame",
            ha="center", va="center", color=FIG_MUTED, fontsize=11, transform=ax.transAxes,
        )
        ax.set_yticks([])
        ax.set_xticks([])

    ax.set_title("Prediction Confidence Scores", color=FIG_TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=FIG_MUTED, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(FIG_GRID)

    # --- title block -----------------------------------------------------
    label = result.get("label", "--")
    accent = CLASS_COLORS.get(label, "#2FBFA8")
    fig.text(0.035, 0.965, f"Result: {label}", color=accent, fontsize=15,
             fontweight="bold", va="top")

    subtitle = result.get("reason") or ""
    if timings and isinstance(timings.get("Total Pipeline"), (int, float)):
        gap = "   ·   " if subtitle else ""
        subtitle = f"{subtitle}{gap}total {timings['Total Pipeline']:.1f} ms"
    if subtitle:
        fig.text(0.035, 0.922, subtitle, color=FIG_MUTED, fontsize=9, va="top")

    fig.subplots_adjust(top=0.85, bottom=0.09, left=0.04, right=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=FIG_BG)
    plt.close(fig)


def _error_payload(message, timings):
    return {
        "label": "ML Error",
        "prediction": "ML Error",
        "probabilities": {},
        "reason": message,
        "details": {"reason": message},
        "timings": timings,
    }


def main() -> None:
    pipeline_t0 = _now_ms()
    timings = {}

    if len(sys.argv) < 2:
        print(json.dumps(_error_payload("No image path given.", timings)))
        return

    image_path = sys.argv[1]

    try:
        model_data, gate_data = _load_models(timings)
    except Exception as exc:
        timings["Total Pipeline"] = round(_now_ms() - pipeline_t0, 2)
        timings["total_ms"] = timings["Total Pipeline"]
        print(json.dumps(_error_payload(f"Could not load Stage D model: {exc}", timings)))
        return

    try:
        result = run_inference(image_path, model_data, gate_data, timings)
    except Exception as exc:
        timings["Total Pipeline"] = round(_now_ms() - pipeline_t0, 2)
        timings["total_ms"] = timings["Total Pipeline"]
        print(json.dumps(_error_payload(f"Inference failed: {exc}", timings)))
        return

    segmentation_paths = save_segmentation_images(result)

    vis_path = None
    t0 = _now_ms()
    try:
        # Provisional total so the figure can print a latency line; the
        # final total below also covers the time spent drawing it.
        timings["Total Pipeline"] = round(_now_ms() - pipeline_t0, 2)
        save_visualization(result, VIS_PATH, timings)
        vis_path = str(VIS_PATH)
    except Exception as viz_err:
        print(f"[predict_runner_stageD] visualization failed: {viz_err}", file=sys.stderr)
    timings["Visualization"] = round(_now_ms() - t0, 2)
    timings["visualization_ms"] = timings["Visualization"]

    timings["Total Pipeline"] = round(_now_ms() - pipeline_t0, 2)
    timings["total_ms"] = timings["Total Pipeline"]

    details = {
        "reason": result["reason"],
        "leaf_ratio": result["leaf_ratio"],
        "component_count": result["component_count"],
        "confidence": result["confidence"],
        "gate_leaf_probability": result["gate_leaf_probability"],
        "model_name": MODEL_DISPLAY_NAME,
        "image_size": IMG_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "gate_threshold": GATE_THRESHOLD,
    }
    # Visual metrics are merged in flat (not nested) because the details
    # window looks them up by bare name, e.g. details["yellow_ratio"].
    details.update(result.get("visual_metrics") or {})
    details.update(result.get("quality") or {})
    details["quality"] = result.get("quality") or {}
    details["quality_thresholds"] = {
        "min_sharpness": MIN_SHARPNESS,
        "max_border_contact": MAX_BORDER_CONTACT,
        "min_solidity": MIN_SOLIDITY,
        "min_largest_region_share": MIN_LARGEST_REGION_SHARE,
    }
    details.update(segmentation_paths)

    payload = {
        "label": result["label"],
        "prediction": result["label"],
        "probabilities": result["probabilities"],
        "reason": result["reason"],
        "details": details,
        "timings": timings,
    }
    payload.update(segmentation_paths)

    if vis_path:
        payload["visualization_path"] = vis_path
        payload["segmentation_image_path"] = vis_path
        payload["details"]["visualization_path"] = vis_path

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
