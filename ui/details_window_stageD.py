"""
ui/details_window_stageD.py

"More Details" popup for the Stage D build of the Kangkong Aquaponics
Monitor.

This is a separate module from the Article 1 baseline's details_window.py
and reads its colours from theme_stageD.py, so both builds can live in the
same project folder without either one changing how the other looks.

Five tabs, in the order someone actually reads them:

    Care          — what to do about the water and the plant
    Segmentation  — the original frame, the leaf mask, the isolated leaf
    Findings      — what the image looked like, in words
    Timing        — per-stage latency
    Inference     — raw model output, thresholds and total latency

Takes the running AquaponicsUI instance so it can read the latest sensor
values and prediction payload without duplicating that state.

Wiring note: the Findings tab reads engineered measurements
(yellow_ratio, lesion_ratio, lesion_count, ...) out of payload["details"].
predict_runner_stageD.py now emits them, which is what the old placeholder
line ("the underlying color/spot signals aren't reaching this window yet")
was waiting for.
"""

import os
import tkinter as tk
from tkinter import ttk

from theme_stageD import (
    COLORS,
    TEMP_LOW_C,
    TEMP_HIGH_C,
    PH_LOW,
    PH_HIGH,
    LIGHT_LOW_LUX,
    make_button,
    shade,
)

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - Pillow is already a UI dependency
    Image = None
    ImageTk = None


# Stage names in pipeline order. The runner emits these directly; the
# aliases map older snake_case keys onto the same rows so a payload from a
# previous build still shows a full breakdown.
STAGE_ORDER = [
    "Model Loading",
    "Image Loading",
    "Scene Scan",
    "Segmentation",
    "Quality Check",
    "Feature Extraction",
    "Visual Metrics",
    "Gate Model",
    "GBM Predict",
    "Predict Probability",
    "Visualization",
]

STAGE_ALIASES = {
    "Model Loading": ["model_load_ms"],
    "Image Loading": ["image_read_ms", "image_load_ms"],
    "Scene Scan": ["component_count_ms"],
    "Segmentation": ["segmentation_ms"],
    "Quality Check": ["quality_check_ms"],
    "Feature Extraction": ["feature_extraction_ms"],
    "Visual Metrics": ["visual_metrics_ms"],
    "Gate Model": ["gate_inference_ms"],
    "GBM Predict": ["predict_ms"],
    "Predict Probability": ["predict_proba_ms", "classification_ms"],
    "Visualization": ["visualization_ms"],
}

TOTAL_KEYS = ["Total Pipeline", "total_ms", "total_pipeline_ms"]

SEGMENTATION_PANELS = [
    ("original_image_path", "Original image", "The captured frame, resized to the analysis size."),
    ("mask_image_path", "Detected leaf mask", "White pixels are the leaf region passed to the model."),
    ("segmented_image_path", "Segmented leaf", "Everything outside the mask is removed before measuring."),
]


# ---------------------------------------------------------------------------
# Payload access
# ---------------------------------------------------------------------------

def _merge_payload(app):
    """Flatten the payload.

    predict_runner_stageD.py nests reason / leaf_ratio / component_count /
    the visual metrics inside payload["details"], while label,
    probabilities and timings live at the top level. Merge them into one
    lookup (nested wins on collisions, being the more specific source) so
    nothing here needs to know which layer a value lives in.
    """
    payload = app.last_prediction_payload if isinstance(app.last_prediction_payload, dict) else {}
    nested = payload.get("details")
    nested = nested if isinstance(nested, dict) else {}
    top_level = {k: v for k, v in payload.items() if k != "details"}
    return {**top_level, **nested}


def _first_present(merged, *keys):
    """First key present with a usable value, else None."""
    for key in keys:
        if key in merged and merged[key] is not None:
            return merged[key]
    return None


def _number(value):
    """Return value as a float if it is numeric, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _resolve_path(app, value):
    """Turn a payload path into something openable, relative paths and all."""
    if not isinstance(value, str) or not value.strip():
        return None
    resolver = getattr(app, "_resolve_possible_path", None)
    if callable(resolver):
        resolved = resolver(value)
        if resolved:
            return resolved
    return value if os.path.exists(value) else None


# ---------------------------------------------------------------------------
# Tab content builders
# ---------------------------------------------------------------------------

def _build_inference_items(app):
    """Raw model output: label, reason, gating numbers, per-class scores,
    and the end-to-end latency for the run that produced them."""
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)

    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    reason = app.last_prediction_reason or merged.get("reason") or "No reason returned."
    leaf_ratio = _number(merged.get("leaf_ratio"))
    components = _number(merged.get("component_count"))
    gate_prob = _number(merged.get("gate_leaf_probability"))
    confidence = _number(merged.get("confidence"))
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}

    accent = app._prediction_text_color(label)
    items = [(f"Prediction: {label}", accent), (f"Reason: {reason}", COLORS["text_muted"])]

    model_name = merged.get("model_name")
    if model_name:
        items.append((f"Model: {model_name}", COLORS["text_muted"]))

    if confidence is not None:
        items.append((f"Confidence: {confidence:.4f}", accent))
    if leaf_ratio is not None:
        items.append((f"Leaf area ratio: {leaf_ratio:.4f}", COLORS["yellow"]))
    if components is not None:
        items.append((f"Leaf-like regions detected: {int(components)}", COLORS["yellow"]))
    if gate_prob is not None:
        items.append((f"Gate leaf probability: {gate_prob:.4f}", COLORS["yellow"]))

    prob_colors = {"Healthy": COLORS["green"], "Discolored": COLORS["yellow"], "Diseased": COLORS["red"]}
    for cls_name in ["Healthy", "Discolored", "Diseased"]:
        value = _number(probs.get(cls_name))
        if value is not None:
            items.append((f"{cls_name} score: {value:.4f}", prob_colors[cls_name]))

    # --- latency, repeated here on purpose ---------------------------------
    # Timing has its own tab, but the headline numbers belong next to the
    # result they describe: this is the tab someone screenshots for a
    # report, and "how fast" is part of the result.
    timings = merged.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    total = _number(_first_present(timings, *TOTAL_KEYS))
    if total is not None:
        items.append((f"Total pipeline latency: {total:.2f} ms", COLORS["teal"]))

    inference_ms = 0.0
    for stage in ("GBM Predict", "Predict Probability"):
        value = _number(_first_present(timings, stage, *STAGE_ALIASES.get(stage, [])))
        if value is not None:
            inference_ms += value
    if inference_ms > 0:
        items.append((f"Model inference latency: {inference_ms:.2f} ms", COLORS["teal"]))

    feature_ms = _number(_first_present(timings, "Feature Extraction", "feature_extraction_ms"))
    if feature_ms is not None:
        items.append((f"Feature extraction latency: {feature_ms:.2f} ms", COLORS["teal"]))

    conf_threshold = _number(merged.get("confidence_threshold"))
    gate_threshold = _number(merged.get("gate_threshold"))
    if conf_threshold is not None:
        items.append((f"Accept threshold: confidence must reach {conf_threshold:.2f}", COLORS["text_muted"]))
    if gate_threshold is not None:
        items.append((f"Gate threshold: leaf probability must reach {gate_threshold:.2f}", COLORS["text_muted"]))

    return items


def _build_visual_findings(app):
    """Plain-language description of what the image itself looked like,
    built from the measurements the runner reports alongside the
    prediction: colour ratios, spot count, spot shape, edge density and
    brightness spread, all measured over leaf pixels only.

    These describe colour and shape. There's no hole or puncture detector
    in the pipeline, so nothing here claims damage of that kind.
    """
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}
    confidence = _number(probs.get(label)) or _number(merged.get("confidence"))

    green_ratio = _number(_first_present(merged, "green_ratio"))
    yellow_ratio = _number(_first_present(merged, "yellow_ratio"))
    lesion_ratio = _number(_first_present(merged, "lesion_ratio"))
    dark_ratio = _number(_first_present(merged, "dark_ratio"))
    lesion_count = _number(_first_present(merged, "lesion_count"))
    mean_circularity = _number(_first_present(merged, "mean_circularity"))
    largest_lesion = _number(_first_present(merged, "largest_lesion_ratio"))
    fragmentation = _number(_first_present(merged, "fragmentation"))
    edge_density = _number(_first_present(merged, "edge_density"))
    std_gray = _number(_first_present(merged, "std_gray"))

    have_metrics = any(
        v is not None for v in
        (green_ratio, yellow_ratio, lesion_ratio, dark_ratio, edge_density, std_gray)
    )

    if not have_metrics:
        conf_text = f" at {confidence * 100:.1f}% confidence" if confidence is not None else ""
        return [(
            f"The frame was classified as {label}{conf_text}, but this run reported no colour or "
            "spot measurements to describe. That happens when the leaf mask came back empty, or "
            "when an older predict_runner_stageD.py produced the result — run ANALYZE again with "
            "the current runner to populate this tab.",
            COLORS["text_muted"],
        )]

    items = []

    # --- overall colour ---------------------------------------------------
    if green_ratio is not None:
        if green_ratio >= 0.6:
            items.append((
                f"Healthy green tissue covers {green_ratio:.1%} of the leaf area — the dominant "
                "colour in this frame.",
                COLORS["green"],
            ))
        elif green_ratio >= 0.3:
            items.append((
                f"Green tissue covers {green_ratio:.1%} of the leaf area. A good part of the leaf "
                "has drifted out of the healthy green range.",
                COLORS["yellow"],
            ))
        else:
            items.append((
                f"Only {green_ratio:.1%} of the leaf area still reads as healthy green.",
                COLORS["red"],
            ))

    if yellow_ratio is not None:
        if yellow_ratio > 0.12:
            items.append((
                f"Yellowing covers {yellow_ratio:.1%} of the leaf — consistent with chlorosis or an "
                "early nutrient deficiency rather than healthy tissue.",
                COLORS["yellow"],
            ))
        elif yellow_ratio > 0.04:
            items.append((
                f"A small yellow-hued area ({yellow_ratio:.1%}) is present. Worth watching, not a "
                "strong signal on its own.",
                COLORS["yellow"],
            ))
        else:
            items.append((
                f"Yellowing is minimal ({yellow_ratio:.1%} of the leaf area).",
                COLORS["green"],
            ))

    # --- spots and necrosis ----------------------------------------------
    if lesion_ratio is not None and lesion_ratio > 0.02:
        detail = f"Brown or necrotic discoloration covers {lesion_ratio:.1%} of the leaf."
        color = COLORS["yellow"]

        if lesion_count is not None and lesion_count >= 1:
            count = int(lesion_count)
            if count >= 3 and mean_circularity is not None and mean_circularity > 0.55:
                detail += (
                    f" It appears as {count} small, roughly circular spots (mean circularity "
                    f"{mean_circularity:.2f}) — a pattern more typical of disease lesions than of "
                    "general discoloration."
                )
                color = COLORS["red"]
            elif count <= 2:
                detail += (
                    f" It appears as {count} larger, less regular patch(es) — more typical of "
                    "general discoloration or chlorosis than of active disease spotting."
                )
            else:
                detail += f" It appears as {count} distinct regions."
                color = COLORS["red"] if count >= 5 else COLORS["yellow"]

        if largest_lesion is not None and largest_lesion > 0.05:
            detail += f" The largest single region alone covers {largest_lesion:.1%} of the leaf."

        items.append((detail, color))
    elif lesion_ratio is not None:
        items.append((
            f"Brown or necrotic discoloration is negligible ({lesion_ratio:.1%} of the leaf area).",
            COLORS["green"],
        ))

    if dark_ratio is not None and dark_ratio > 0.03:
        items.append((
            f"{dark_ratio:.1%} of the leaf reads as very dark or near-black, which usually marks "
            "genuinely dead tissue at the centre of a lesion rather than simple discoloration.",
            COLORS["red"],
        ))

    if fragmentation is not None and fragmentation > 8 and lesion_ratio and lesion_ratio > 0.02:
        items.append((
            f"The affected area is highly fragmented ({fragmentation:.1f} spots per 1% of leaf "
            "area) — many small scattered spots rather than one stress patch, which points toward "
            "active disease.",
            COLORS["red"],
        ))

    # --- surface texture --------------------------------------------------
    if edge_density is not None:
        if edge_density > 0.15:
            items.append((
                f"Edge density is high ({edge_density:.3f}): the surface has sharp, high-contrast "
                "boundaries, the signature of distinct spots rather than a smooth colour gradient.",
                COLORS["yellow"],
            ))
        else:
            items.append((
                f"Edge density is low ({edge_density:.3f}): colour changes across the leaf are "
                "gradual rather than sharply bounded.",
                COLORS["green"],
            ))

    if std_gray is not None and std_gray > 55:
        items.append((
            f"Brightness across the leaf is uneven (spread {std_gray:.1f}), which reads as mottling "
            "— though strong shadow or glare in the frame can produce the same number.",
            COLORS["yellow"],
        ))

    conf_text = f" at {confidence * 100:.1f}% confidence" if confidence is not None else ""
    items.append((
        f"Taken together, the model classified this frame as {label}{conf_text}.",
        app._prediction_text_color(label),
    ))

    return items


def _build_quality_items(app):
    """How good the capture itself was.

    These come from the runner's quality gate. They're shown whether the
    frame passed or failed: on a pass they're reassurance that the result
    rests on a usable photo, and on a fail they say exactly which check
    stopped it and by how much.
    """
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    thresholds = merged.get("quality_thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}

    sharpness = _number(_first_present(merged, "sharpness", "frame_sharpness"))
    border = _number(merged.get("border_contact"))
    solidity = _number(merged.get("solidity"))
    largest_share = _number(merged.get("largest_region_share"))
    brightness = _number(merged.get("mean_brightness"))
    overexposed = _number(merged.get("overexposed_ratio"))
    underexposed = _number(merged.get("underexposed_ratio"))

    if all(v is None for v in (sharpness, border, solidity, largest_share, brightness)):
        return None

    min_sharpness = _number(thresholds.get("min_sharpness")) or 55.0
    max_border = _number(thresholds.get("max_border_contact")) or 0.25
    min_solidity = _number(thresholds.get("min_solidity")) or 0.70
    min_share = _number(thresholds.get("min_largest_region_share")) or 0.65

    items = []

    if sharpness is not None:
        ok = sharpness >= min_sharpness
        items.append((
            f"Sharpness {sharpness:.0f} (needs {min_sharpness:.0f}) — "
            + ("in focus." if ok else "too blurry to analyze; hold the camera still and let it focus."),
            COLORS["green"] if ok else COLORS["red"],
        ))

    if border is not None:
        ok = border <= max_border
        items.append((
            f"Framing: the leaf covers {border:.0%} of the frame border (limit {max_border:.0%}) — "
            + ("it sits inside the frame." if ok else "it runs off the edge; move the camera back."),
            COLORS["green"] if ok else COLORS["red"],
        ))

    if largest_share is not None:
        ok = largest_share >= min_share
        items.append((
            f"Isolation: the main region is {largest_share:.0%} of the detected area "
            f"(needs {min_share:.0%}) — "
            + ("one leaf dominates the frame." if ok else "several separate regions are in view."),
            COLORS["green"] if ok else COLORS["red"],
        ))

    if solidity is not None:
        ok = solidity >= min_solidity
        items.append((
            f"Shape: solidity {solidity:.2f} (needs {min_solidity:.2f}) — "
            + ("compact, leaf-like outline." if ok else
               "ragged outline; overlapping stems or background foliage are being picked up."),
            COLORS["green"] if ok else COLORS["red"],
        ))

    if brightness is not None:
        glare = overexposed if overexposed is not None else 0.0
        shadow = underexposed if underexposed is not None else 0.0
        if glare > 0.25:
            items.append((f"Exposure: {glare:.0%} of the leaf is blown out by glare.", COLORS["red"]))
        elif shadow > 0.35:
            items.append((f"Exposure: {shadow:.0%} of the leaf is in deep shadow.", COLORS["red"]))
        else:
            items.append((f"Exposure: mean leaf brightness {brightness:.0f} — well lit.", COLORS["green"]))

    return items


def _build_performance_items(app):
    """Per-stage timing breakdown from the last analysis run."""
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    timings = merged.get("timings")
    if not isinstance(timings, dict) or not timings:
        return None

    items = []
    measured_total = 0.0
    for stage in STAGE_ORDER:
        value = _number(_first_present(timings, stage, *STAGE_ALIASES.get(stage, [])))
        if value is None:
            continue
        measured_total += value
        items.append((f"{stage}: {value:.2f} ms", COLORS["text"]))

    if not items:
        return None

    # Prefer the runner's own wall-clock total: it includes the glue
    # between stages, so it's always >= the sum of the rows above.
    total = _number(_first_present(timings, *TOTAL_KEYS))
    if total is None:
        total = measured_total
    items.append((f"Total pipeline: {total:.2f} ms", COLORS["teal"]))

    overhead = total - measured_total
    if overhead > 1.0:
        items.append((
            f"Unaccounted overhead: {overhead:.2f} ms (process start-up and glue between stages)",
            COLORS["text_muted"],
        ))

    return items


def _build_segmentation_panels(app):
    """Paths + captions for the three segmentation stage images, plus a
    line of mask statistics. Returns None when the runner didn't write
    them (older build, or the figure step failed)."""
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    panels = []
    for key, title, caption in SEGMENTATION_PANELS:
        path = _resolve_path(app, merged.get(key))
        if path:
            panels.append({"path": path, "title": title, "caption": caption})

    if not panels:
        return None

    leaf_ratio = _number(merged.get("leaf_ratio"))
    components = _number(merged.get("component_count"))
    stats = []
    if leaf_ratio is not None:
        stats.append(f"leaf covers {leaf_ratio:.2%} of the frame")
    if components is not None:
        stats.append(f"{int(components)} leaf-like region(s) found")

    return {"panels": panels, "stats": "  ·  ".join(stats)}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _rec_severity_color(text: str) -> str:
    lowered = text.lower()
    if "optimal" in lowered or "good light" in lowered or "keep up" in lowered:
        return COLORS["green"]
    if "unavailable" in lowered or "no data" in lowered:
        return COLORS["text_muted"]
    if any(phrase in lowered for phrase in
           ("too cold", "too hot", "too acidic", "too alkaline", "too low", "disease")):
        return COLORS["red"]
    return COLORS["yellow"]


def _rec_severity_icon(text: str) -> str:
    lowered = text.lower()
    if "optimal" in lowered or "good light" in lowered or "keep up" in lowered:
        return "\N{CHECK MARK}"
    if "unavailable" in lowered or "no data" in lowered:
        return "\N{BULLET}"
    if any(phrase in lowered for phrase in
           ("too cold", "too hot", "too acidic", "too alkaline", "too low", "disease")):
        return "\N{HEAVY BALLOT X}"
    return "\N{WARNING SIGN}"


def _build_plant_health_recommendation(app):
    """What the last analysis found, plus the matching care advice."""
    if not app.last_prediction_payload:
        return "Plant health: capture and analyze an image to get a recommendation."

    merged = _merge_payload(app)
    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    reason = app.last_prediction_reason or merged.get("reason")
    leaf_ratio = _number(merged.get("leaf_ratio"))
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}
    confidence = _number(probs.get(label)) or _number(merged.get("confidence"))

    findings = []
    if confidence is not None:
        findings.append(f"{confidence * 100:.1f}% confidence")
    if reason:
        findings.append(str(reason))
    if leaf_ratio is not None:
        findings.append(f"leaf coverage {leaf_ratio:.1%}")
    findings_text = f" ({'; '.join(findings)})" if findings else ""

    if label == "Diseased":
        return (
            f"Plant health: the last analysis found signs of disease{findings_text}. "
            "Check for pests, root rot and nutrient deficiency (nitrogen or iron), and "
            "isolate the affected plant."
        )
    if label == "Discolored":
        return (
            f"Plant health: the last analysis found discoloration{findings_text}. "
            "This often means early nutrient deficiency or light stress — review the recent "
            "pH, temperature and lux readings and watch whether it spreads."
        )
    if label == "Healthy":
        return (
            f"Plant health: the last analysis found the leaf healthy{findings_text}. "
            "Keep up the current watering, lighting and nutrient routine."
        )

    return f"Plant health: the last analysis returned '{label}'{findings_text}."


def _build_recommendations(app):
    lux = app.last_sensor_values.get("lux")
    temp = app.last_sensor_values.get("temp_c")
    ph = app.last_sensor_values.get("ph")

    recs = []

    if temp is None:
        recs.append("Water temperature: no data available yet.")
    elif temp < TEMP_LOW_C:
        recs.append(
            f"Water temperature ({temp:.2f} °C): too cold. Kangkong prefers "
            f"{TEMP_LOW_C}-{TEMP_HIGH_C} °C — add an aquarium heater."
        )
    elif temp > TEMP_HIGH_C:
        recs.append(
            f"Water temperature ({temp:.2f} °C): too hot. Do a partial water change with cooler "
            "water, add shade, or improve ventilation."
        )
    else:
        recs.append(f"Water temperature ({temp:.2f} °C): optimal for Kangkong.")

    if ph is None:
        recs.append("pH level: no data available yet.")
    elif ph < PH_LOW:
        recs.append(
            f"pH level ({ph:.2f}): too acidic. Add pH UP solution or agricultural lime to bring it "
            f"back to {PH_LOW}-{PH_HIGH}."
        )
    elif ph > PH_HIGH:
        recs.append(
            f"pH level ({ph:.2f}): too alkaline. Add pH DOWN solution (phosphoric acid) to bring it "
            f"back to {PH_LOW}-{PH_HIGH}."
        )
    else:
        recs.append(f"pH level ({ph:.2f}): optimal — this range absorbs nutrients well.")

    if lux is None:
        recs.append("Light: no data available yet.")
    elif lux < LIGHT_LOW_LUX:
        recs.append(
            f"Light ({lux:.2f} lux): too low. Turn on the grow lights or move the setup somewhere "
            "sunnier."
        )
    else:
        recs.append(f"Light ({lux:.2f} lux): good light intensity — the plants have enough.")

    recs.append(_build_plant_health_recommendation(app))
    return recs


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------

def _add_card(parent, text, color, icon=None, wraplength=640):
    """One finding/recommendation row: accent strip, optional icon, text."""
    outer = tk.Frame(parent, bg=COLORS["bg"])
    outer.pack(fill=tk.X, pady=3)

    card = tk.Frame(
        outer, bg=COLORS["bg_card"],
        highlightbackground=COLORS["border_soft"], highlightthickness=1, bd=0,
    )
    card.pack(fill=tk.X)

    tk.Frame(card, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)

    inner = tk.Frame(card, bg=COLORS["bg_card"])
    inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 12), pady=9)

    if icon:
        tk.Label(
            inner, text=icon, font=("Helvetica", 11, "bold"),
            bg=COLORS["bg_card"], fg=color,
        ).pack(side=tk.LEFT, padx=(0, 9), anchor="n")

    tk.Label(
        inner, text=text, font=("Helvetica", 10), bg=COLORS["bg_card"], fg=COLORS["text"],
        wraplength=wraplength, justify="left", anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Subtle hover tint so a long list doesn't read as a flat wall.
    hover = shade(COLORS["bg_card"], 9)

    def repaint(bg):
        card.configure(bg=bg)
        inner.configure(bg=bg)
        for child in inner.winfo_children():
            child.configure(bg=bg)

    card.bind("<Enter>", lambda _e: repaint(hover))
    card.bind("<Leave>", lambda _e: repaint(COLORS["bg_card"]))

    return card


def _add_note(parent, text):
    tk.Label(
        parent, text=text, font=("Helvetica", 9), bg=COLORS["bg"], fg=COLORS["text_muted"],
        wraplength=700, justify="left", anchor="w",
    ).pack(fill=tk.X, pady=(0, 8))


def _add_tab_panel(notebook, label):
    """One notebook tab with its own scrollable, padded content area.

    Returns (content_frame, unbind_wheel): pack into content_frame, and
    call unbind_wheel() on close so this panel's bind_all() mousewheel
    handler doesn't outlive the window.
    """
    page = tk.Frame(notebook, bg=COLORS["bg"])
    notebook.add(page, text=label)

    canvas = tk.Canvas(page, bg=COLORS["bg"], highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_y = tk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scroll_y.set)

    inner = tk.Frame(canvas, bg=COLORS["bg"])
    canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(canvas_window, width=e.width))

    def on_mousewheel(event):
        canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", on_mousewheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

    content = tk.Frame(inner, bg=COLORS["bg"])
    content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 14))

    return content, (lambda: canvas.unbind_all("<MouseWheel>"))


def _add_segmentation_panels(parent, segmentation, image_refs, thumb=190):
    """Three side-by-side image panels showing the segmentation stages.

    Tk drops an image the moment nothing references it, so every
    PhotoImage is appended to image_refs, which the caller keeps alive for
    the lifetime of the window.
    """
    if Image is None or ImageTk is None:
        _add_card(parent, "Pillow isn't available, so the segmentation images can't be "
                          "displayed here.", COLORS["text_muted"])
        return

    if segmentation.get("stats"):
        _add_note(parent, segmentation["stats"])

    row = tk.Frame(parent, bg=COLORS["bg"])
    row.pack(fill=tk.X)

    for panel in segmentation["panels"]:
        column = tk.Frame(
            row, bg=COLORS["bg_card"],
            highlightbackground=COLORS["border_soft"], highlightthickness=1, bd=0,
        )
        column.pack(side=tk.LEFT, padx=(0, 8), pady=2, anchor="n")

        tk.Label(
            column, text=panel["title"], font=("Helvetica", 10, "bold"),
            bg=COLORS["bg_card"], fg=COLORS["text"], anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(9, 6))

        holder = tk.Label(column, bg=COLORS["bg"], bd=0)
        holder.pack(padx=10)

        try:
            image = Image.open(panel["path"]).convert("RGB")
            image.thumbnail((thumb, thumb), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            image_refs.append(photo)
            holder.configure(image=photo, width=image.width, height=image.height)
        except Exception as exc:
            holder.configure(
                text=f"Could not load\n{os.path.basename(panel['path'])}\n{exc}",
                fg=COLORS["text_muted"], font=("Helvetica", 9),
                width=24, height=8, wraplength=thumb,
            )

        tk.Label(
            column, text=panel["caption"], font=("Helvetica", 9),
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            wraplength=thumb, justify="left", anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(7, 10))


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def open_recommendations_window(app):
    """Build and show the analysis details popup.

    `app` is the running AquaponicsUI instance (needs .root, .lbl_status,
    .last_sensor_values and the last_prediction_* attributes).
    """
    window = tk.Toplevel(app.root)
    window.title("Analysis details")
    # Sized to fit inside the 800x480 panel rather than overflowing it,
    # which is what cut the old 760x600 window off at the bottom.
    window.geometry("784x466")
    window.minsize(600, 380)
    window.configure(bg=COLORS["bg"])
    window.transient(app.root)

    image_refs = []   # keeps segmentation thumbnails from being garbage collected

    # --- header -----------------------------------------------------------
    header = tk.Frame(window, bg=COLORS["bg_panel"])
    header.pack(fill=tk.X)

    header_text = tk.Frame(header, bg=COLORS["bg_panel"])
    header_text.pack(side=tk.LEFT, padx=16, pady=10)

    label = app.last_prediction_label or "No result yet"
    tk.Label(
        header_text, text=label, font=("Helvetica", 16, "bold"),
        bg=COLORS["bg_panel"], fg=app._prediction_text_color(label),
    ).pack(anchor="w")
    tk.Label(
        header_text, text="Image analysis, sensor status and care recommendations",
        font=("Helvetica", 9), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
    ).pack(anchor="w")

    close_slot = {}
    make_button(
        header, "CLOSE", COLORS["red"], COLORS["red_hover"],
        lambda: close_slot["close"](), font=("Helvetica", 10, "bold"),
    ).pack(side=tk.RIGHT, padx=16)

    tk.Frame(window, bg=COLORS["border"], height=1).pack(fill=tk.X)

    # --- content ----------------------------------------------------------
    recs = _build_recommendations(app)
    segmentation = _build_segmentation_panels(app)
    visual_items = _build_visual_findings(app)
    performance_items = _build_performance_items(app)
    inference_items = _build_inference_items(app)

    body = tk.Frame(window, bg=COLORS["bg"])
    body.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure("Details.TNotebook", background=COLORS["bg"], borderwidth=0,
                    tabmargins=(0, 0, 0, 0))
    style.configure(
        "Details.TNotebook.Tab",
        background=COLORS["bg_panel"], foreground=COLORS["text_muted"],
        padding=(14, 8), font=("Helvetica", 10, "bold"), borderwidth=0,
    )
    style.map(
        "Details.TNotebook.Tab",
        background=[("selected", COLORS["bg_card"])],
        foreground=[("selected", COLORS["text"])],
    )

    notebook = ttk.Notebook(body, style="Details.TNotebook")
    notebook.pack(fill=tk.BOTH, expand=True)

    wheel_unbinders = []

    def new_tab(title):
        panel, unbind = _add_tab_panel(notebook, title)
        wheel_unbinders.append(unbind)
        return panel

    # Care first — it's why someone opens this window.
    care_panel = new_tab("Care")
    for rec in recs:
        _add_card(care_panel, rec, _rec_severity_color(rec), icon=_rec_severity_icon(rec))

    if segmentation:
        seg_panel = new_tab("Segmentation")
        _add_segmentation_panels(seg_panel, segmentation, image_refs)
    elif app.last_prediction_payload:
        seg_panel = new_tab("Segmentation")
        _add_card(
            seg_panel,
            "This run didn't write the segmentation images. Run ANALYZE again with the current "
            "predict_runner_stageD.py, which saves the mask and the isolated leaf to ml/outputs/.",
            COLORS["text_muted"],
        )

    quality_items = _build_quality_items(app)
    if visual_items or quality_items:
        findings_panel = new_tab("Findings")
        if quality_items:
            _add_note(findings_panel, "Capture quality — every check below has to pass before the "
                                      "model is allowed to classify the frame.")
            for text, color in quality_items:
                _add_card(findings_panel, text, color)
        if visual_items:
            _add_note(findings_panel, "What the leaf looks like. Measured over leaf pixels only, "
                                      "so the background can't affect these numbers.")
            for text, color in visual_items:
                _add_card(findings_panel, text, color)

    if performance_items:
        timing_panel = new_tab("Timing")
        for text, color in performance_items:
            _add_card(timing_panel, text, color)

    if inference_items:
        inference_panel = new_tab("Inference")
        for text, color in inference_items:
            _add_card(inference_panel, text, color)

    notebook.select(0)

    # --- lifecycle --------------------------------------------------------
    def close():
        for unbind in wheel_unbinders:
            try:
                unbind()
            except Exception:
                pass
        image_refs.clear()
        try:
            window.grab_release()
        except Exception:
            pass
        window.destroy()

    close_slot["close"] = close
    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Escape>", lambda _e: close())

    # Make sure the popup actually receives clicks/focus (some window
    # managers and kiosk sessions otherwise leave CLOSE looking dead).
    window.update_idletasks()
    window.lift()
    window.focus_force()
    window.grab_set()

    return window
