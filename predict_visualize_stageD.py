"""
Paper-ready prediction visualization for the Stage D (Lean) GBM model.

Same inference pipeline as your existing predict_and_visualize, but:
- saves a high-resolution figure to disk (PNG + PDF) instead of only plt.show()
- cleaner, print-friendly labels/typography for direct use in the paper
- keeps the interactive plt.show() too, so you can still preview it locally

Usage:
    Paste this into your project (same folder as feature_extractor.py and
    the models/ directory), then run:

        python predict_visualize_stageD.py

    Edit TEST_IMAGE below to point at the leaf image you want to feature
    as the example figure in the paper.
"""

from pathlib import Path
import time
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature_extractor_stageD_lean import extract_features_from_bgr, segment_leaf
from feature_extractor import count_leaflike_components  # scene-level check unchanged from baseline

MODEL_PATH = Path("models/gbm_model_cv_stageD_lean.pkl")
GATE_MODEL_PATH = Path("models/gate_model.pkl")
IMG_SIZE = 128

MIN_LEAF_RATIO = 0.03
MAX_LEAF_RATIO = 0.65
CONFIDENCE_THRESHOLD = 0.5
GATE_THRESHOLD = 0.5

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "discolored": "Discolored",
    "diseased": "Diseased",
}

# ---- Edit this to the image you want to feature in the paper ----
TEST_IMAGE = "test/test (80).jpg"

# ---- Output settings ----
OUTPUT_DIR = Path("figures")
OUTPUT_BASENAME = "stageD_prediction_example"  # produces .png and .pdf
FIGURE_DPI = 300


def _load_models():
    model_data = joblib.load(MODEL_PATH)
    gate_data = joblib.load(GATE_MODEL_PATH) if GATE_MODEL_PATH.exists() else None
    return model_data, gate_data


def run_inference(image_path, model_data, gate_data,
                   min_leaf_ratio=MIN_LEAF_RATIO,
                   max_leaf_ratio=MAX_LEAF_RATIO,
                   confidence_threshold=CONFIDENCE_THRESHOLD):
    timings = {}
    model = model_data["model"]
    class_names = model_data["class_names"]
    gate_model = gate_data["model"] if gate_data else None

    t0 = time.perf_counter()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        timings["Image Loading"] = (time.perf_counter() - t0) * 1000
        return {
            "result_text": "No clear kangkong leaf detected",
            "reason_text": "Image could not be read.",
            "probs": None, "pred_label": None, "gate_leaf_prob": None,
            "class_names": None, "mask": None, "leaf_ratio": None,
            "scene_component_count": None, "image_rgb": None, "segmented_rgb": None,
            "timings": timings,
        }
    image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    timings["Image Loading"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scene_component_count = count_leaflike_components(image_bgr)
    mask, leaf_ratio, seg_rejected = segment_leaf(image_bgr)
    timings["Segmentation"] = (time.perf_counter() - t0) * 1000

    segmented_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    result_text = "No clear kangkong leaf detected"
    reason_text = ""
    probs = None
    pred_label = None
    gate_leaf_prob = None

    if scene_component_count == 0:
        reason_text = "No leaf-like region detected."
    elif scene_component_count > 1:
        reason_text = f"Multiple leaf-like regions detected ({scene_component_count}). Please isolate one leaf."
    elif seg_rejected:
        reason_text = f"Leaf region rejected by segmentation (leaf ratio={leaf_ratio:.4f})."
    elif leaf_ratio < min_leaf_ratio:
        reason_text = f"Leaf area too small ({leaf_ratio:.4f})"
    elif leaf_ratio > max_leaf_ratio:
        reason_text = f"Scene too cluttered or mask too large ({leaf_ratio:.4f}). Please isolate one leaf."
    else:
        t0 = time.perf_counter()
        features = extract_features_from_bgr(image_bgr).reshape(1, -1)
        timings["Feature Extraction"] = (time.perf_counter() - t0) * 1000

        gate_ok = True
        if gate_model is not None:
            t0 = time.perf_counter()
            gate_leaf_prob = float(gate_model.predict_proba(features)[0][1])
            timings["Gate Model"] = (time.perf_counter() - t0) * 1000
            gate_ok = gate_leaf_prob >= GATE_THRESHOLD
            if not gate_ok:
                reason_text = f"Rejected by gate model (leaf probability={gate_leaf_prob:.4f})"

        if gate_ok:
            t0 = time.perf_counter()
            pred_idx = model.predict(features)[0]
            timings["GBM Predict"] = (time.perf_counter() - t0) * 1000
            pred_label = class_names[pred_idx]

            if hasattr(model, "predict_proba"):
                t0 = time.perf_counter()
                probs = model.predict_proba(features)[0]
                timings["Predict Probability"] = (time.perf_counter() - t0) * 1000
                max_prob = float(np.max(probs))

                if max_prob < confidence_threshold:
                    reason_text = f"Low model confidence ({max_prob:.4f})"
                else:
                    result_text = DISPLAY_NAMES.get(pred_label, pred_label)
                    reason_text = f"Accepted prediction (confidence={max_prob:.4f})"
            else:
                result_text = DISPLAY_NAMES.get(pred_label, pred_label)
                reason_text = "Prediction made without probability output"

    return {
        "result_text": result_text,
        "reason_text": reason_text,
        "probs": probs,
        "pred_label": pred_label,
        "gate_leaf_prob": gate_leaf_prob,
        "class_names": class_names,
        "mask": mask,
        "leaf_ratio": leaf_ratio,
        "scene_component_count": scene_component_count,
        "image_rgb": image_rgb,
        "segmented_rgb": segmented_rgb,
        "timings": timings,
    }


def predict_and_export(image_path,
                        output_dir=OUTPUT_DIR,
                        output_basename=OUTPUT_BASENAME,
                        dpi=FIGURE_DPI,
                        show=True):
    """
    Runs inference and produces a print-quality 4-panel figure:
    original image, leaf mask, segmented region, and prediction
    confidence bar chart. Saves both PNG and PDF versions to output_dir.
    """
    model_data, gate_data = _load_models()
    out = run_inference(image_path, model_data, gate_data)

    print(f"Result: {out['result_text']}")
    print(f"Reason: {out['reason_text']}")
    if out["probs"] is not None:
        print("\nConfidence scores:")
        for label, prob in zip(out["class_names"], out["probs"]):
            print(f"  {DISPLAY_NAMES.get(label, label)}: {prob:.4f}")

    if out["image_rgb"] is None:
        print("Image failed to load — nothing to plot.")
        return out

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 13,
    })

    fig = plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(out["image_rgb"])
    ax1.set_title("(a) Original Image")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(out["mask"], cmap="gray")
    ax2.set_title(f"(b) Leaf Mask (ratio={out['leaf_ratio']:.3f})")
    ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(out["segmented_rgb"])
    ax3.set_title("(c) Segmented Leaf Region")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    if out["probs"] is not None:
        labels = [DISPLAY_NAMES.get(lbl, lbl) for lbl in out["class_names"]]
        bars = ax4.bar(labels, out["probs"], color=["#4C956C", "#F4A259", "#BC4749"])
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("Confidence")
        ax4.set_title("(d) Prediction Confidence")
        for bar, prob in zip(bars, out["probs"]):
            ax4.text(bar.get_x() + bar.get_width() / 2, prob + 0.02,
                      f"{prob:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        skip_reason = "Classification skipped"
        if out["gate_leaf_prob"] is not None:
            skip_reason += f"\n(gate leaf probability={out['gate_leaf_prob']:.3f})"
        ax4.text(0.5, 0.5, skip_reason, ha="center", va="center")
        ax4.set_title("(d) Prediction Confidence")
        ax4.axis("off")

    fig.suptitle(f"Stage D Prediction: {out['result_text']}\n{out['reason_text']}")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_basename}.png"
    pdf_path = output_dir / f"{output_basename}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved figure to:\n  {png_path}\n  {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out


if __name__ == "__main__":
    predict_and_export(TEST_IMAGE)