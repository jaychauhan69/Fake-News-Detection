"""
Image Authenticity Detection System — Orchestrator
=====================================================
Usage:
    python3 analyze.py <path_to_image> [output_dir]

Produces:
    - JSON forensic report
    - Heatmap PNG (suspicious regions highlighted)
    - Human-readable console summary
"""

import sys
import os
import json
import numpy as np

from forensics_core import (
    extract_metadata, error_level_analysis, noise_analysis,
    jpeg_artifact_analysis, color_histogram_analysis,
    lighting_edge_consistency, ai_generation_heuristic,
    run_deep_model_stub, build_heatmap,
)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def analyze_image(path, out_dir="."):
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXT)}")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    meta = extract_metadata(path)
    ela = error_level_analysis(path)
    noise = noise_analysis(path)
    jpeg_art = jpeg_artifact_analysis(path)
    color = color_histogram_analysis(path)
    lighting = lighting_edge_consistency(path)
    ai_heur = ai_generation_heuristic(path)
    deep_model_result = run_deep_model_stub(path)  # None unless a real model is wired in

    heatmap_path = os.path.join(out_dir, f"{base}_heatmap.png")
    heat = build_heatmap(path, ela["ela_map"], noise["noise_map"], heatmap_path)

    scores, indicators, explanation = fuse_scores(
        meta, ela, noise, jpeg_art, color, lighting, ai_heur, deep_model_result, heat
    )

    report = {
        "file": os.path.basename(path),
        "classification": scores["classification"],
        "confidence_percentages": scores["percentages"],
        "metadata_summary": {
            "camera_make": meta.get("camera_make"),
            "camera_model": meta.get("camera_model"),
            "lens_model": meta.get("lens_model"),
            "capture_datetime": meta.get("capture_datetime"),
            "gps": meta.get("gps"),
            "software_tag": meta.get("software"),
            "resolution": meta.get("resolution"),
            "exif_present": meta.get("exif_present"),
            "color_profile": meta.get("color_profile"),
        },
        "editing_software_detected": meta.get("editing_software_detected"),
        "ai_generator_signature_detected": meta.get("ai_generator_detected"),
        "forensic_indicators": indicators,
        "technical_measurements": {
            "ela_mean_error": round(ela["mean_error"], 3),
            "ela_outlier_block_ratio": round(ela["outlier_block_ratio"], 4),
            "noise_global_energy": round(noise["global_noise_energy"], 3),
            "noise_consistency_cv": round(noise["noise_consistency_cv"], 4),
            "jpeg_blockiness_score": jpeg_art["blockiness_score"],
            "likely_recompressed": jpeg_art["likely_recompressed"],
            "saturation_mean": color["saturation_mean"],
            "contrast_std": color["contrast_std"],
            "comb_pattern_detected": color["comb_pattern_detected"],
            "lighting_variance_score": lighting.get("lighting_variance_score"),
            "ai_heuristic_score": ai_heur["ai_heuristic_score"],
        },
        "heatmap_file": os.path.basename(heat["heatmap_path"]),
        "suspicious_regions_bbox_xywh": heat["suspicious_regions"],
        "explanation": explanation,
        "disclosure": (
            "The AI-generation signal in this report comes from a hand-tuned "
            "frequency/noise heuristic, not a trained EfficientNet-B4/ViT "
            "classifier (no labeled training data / model weights available "
            "in this environment). See forensics_core.run_deep_model_stub() "
            "for the integration point to plug in a real trained model."
        ),
    }

    report_path = os.path.join(out_dir, f"{base}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report, report_path, heat["heatmap_path"]


def fuse_scores(meta, ela, noise, jpeg_art, color, lighting, ai_heur, deep_model_result, heat):
    indicators = []

    # ---- Manipulation / editing evidence ----
    edit_score = 0.0
    if ela["outlier_block_ratio"] > 0.02:
        edit_score += 35 * min(ela["outlier_block_ratio"] / 0.1, 1.0)
        indicators.append(f"ELA found {ela['outlier_block_ratio']*100:.1f}% of blocks with anomalous error levels (possible localized edits)")
    if noise["noise_consistency_cv"] > 0.6:
        edit_score += 25 * min((noise["noise_consistency_cv"] - 0.6) / 0.8, 1.0)
        indicators.append(f"Noise pattern inconsistent across regions (CV={noise['noise_consistency_cv']:.2f}), suggesting possible splicing/compositing")
    if jpeg_art["likely_recompressed"]:
        edit_score += 15
        indicators.append(f"JPEG blockiness pattern (score={jpeg_art['blockiness_score']}) suggests re-compression after editing")
    if lighting.get("lighting_variance_score", 0) > 0.5 and not lighting.get("insufficient_texture", True):
        edit_score += 15
        indicators.append(f"Elevated lighting-direction variance ({lighting['lighting_variance_score']:.2f}) across image regions, possibly inconsistent lighting/shadows")
    if meta.get("editing_software_detected"):
        edit_score += 20
        indicators.append(f"Editing software signature found in file: {meta['editing_software_detected']}")
    edit_score = min(edit_score, 100)

    # ---- Color grading evidence ----
    grade_score = 0.0
    if color["excessive_saturation"]:
        grade_score += 30
        indicators.append(f"Elevated/unusual saturation statistics (mean={color['saturation_mean']}, std={color['saturation_std']})")
    if color["comb_pattern_detected"]:
        grade_score += 35
        indicators.append(f"Histogram 'comb' pattern detected (score={color['comb_pattern_score']}), typical of curves/levels/LUT remapping")
    if color["excessive_clipping"]:
        grade_score += 20
        indicators.append(f"Histogram clipping at extremes (avg clip ratio={color['avg_clip_ratio']}), suggesting contrast push")
    if color["high_contrast_push"]:
        grade_score += 15
        indicators.append(f"High global contrast ({color['contrast_std']:.1f}) beyond typical unedited range")
    grade_score = min(grade_score, 100)

    # ---- AI-generation evidence ----
    if deep_model_result is not None:
        ai_score = deep_model_result * 100
        indicators.append("Deep-learning model score used for AI-generation likelihood")
    else:
        ai_score = ai_heur["ai_heuristic_score"] * 100
        indicators.append(
            f"Frequency/noise heuristic AI-likelihood={ai_score:.0f}/100 "
            f"(HF energy ratio={ai_heur['hf_energy_ratio']}, spectral spikiness={ai_heur['spectral_spikiness']}, "
            f"residual noise energy={ai_heur['residual_noise_energy']}) — heuristic, not a trained classifier"
        )
    if meta.get("ai_generator_detected"):
        ai_score = max(ai_score, 90)
        indicators.append(f"AI generator metadata signature found: {meta['ai_generator_detected']}")
    if not meta.get("exif_present") and ai_score > 40:
        ai_score = min(ai_score + 10, 100)
        indicators.append("No camera EXIF metadata present, consistent with AI-generated or metadata-stripped content")

    # ---- Originality evidence (inverse signal) ----
    orig_score = 0.0
    if meta.get("exif_present") and meta.get("camera_model"):
        orig_score += 40
        indicators.append(f"Camera EXIF present: {meta.get('camera_make','?')} {meta.get('camera_model','?')}")
    if noise["global_noise_energy"] > 3.0 and noise["noise_consistency_cv"] < 0.6:
        orig_score += 25
        indicators.append(f"Consistent sensor-like noise pattern across image (energy={noise['global_noise_energy']:.2f})")
    if not meta.get("editing_software_detected") and not meta.get("ai_generator_detected"):
        orig_score += 15
    if ela["outlier_block_ratio"] < 0.01:
        orig_score += 20
    orig_score = min(orig_score, 100)

    raw = {
        "Original": orig_score,
        "AI Generated": ai_score,
        "Edited/Manipulated": edit_score,
        "Color Graded/Enhanced": grade_score,
    }

    # Normalize to percentages that sum to 100 (soft fusion, not mutually exclusive
    # ground truth — an image can be both edited AND color graded, but we report
    # a primary classification = highest scoring category)
    total = sum(raw.values())
    if total <= 1e-6:
        percentages = {k: 25.0 for k in raw}
    else:
        # apply mild softmax-like sharpening so a clear winner isn't diluted
        arr = np.array(list(raw.values()), dtype=np.float64)
        arr = np.power(arr + 1.0, 1.4)
        arr = arr / arr.sum() * 100
        percentages = {k: round(float(v), 1) for k, v in zip(raw.keys(), arr)}

    classification = max(percentages, key=percentages.get)

    explanation = build_explanation(classification, percentages, indicators)

    return {"classification": classification, "percentages": percentages}, indicators, explanation


def build_explanation(classification, percentages, indicators):
    top_reasons = indicators[:5] if indicators else ["No strong forensic signals were detected in any direction."]
    lines = [
        f"This image was classified as '{classification}' with {percentages[classification]}% confidence, "
        f"based on fusing metadata, Error Level Analysis, sensor-noise consistency, JPEG artifact analysis, "
        f"color-histogram/LUT analysis, lighting consistency, and a frequency-based AI-generation heuristic.",
        "Key contributing indicators:",
    ]
    lines += [f"  - {r}" for r in top_reasons]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <image_path> [output_dir]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    report, report_path, heatmap_path = analyze_image(img_path, out_dir)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to: {report_path}")
    print(f"Heatmap saved to: {heatmap_path}")
