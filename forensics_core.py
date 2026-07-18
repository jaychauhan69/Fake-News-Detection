"""
Image Authenticity Detection System — Core Engine
====================================================
Combines classical forensic techniques with a frequency/statistical
"AI-generation likelihood" heuristic to classify an image as:
    Original / AI-Generated / Edited-Manipulated / Color-Graded

IMPORTANT HONESTY NOTE (read this):
This module performs REAL, working analysis for steps 1-6 and 8-9
(EXIF, ELA, noise/sensor pattern analysis, JPEG artifact/blockiness
detection, color-histogram/LUT analysis, lighting/edge consistency,
heatmap generation, weighted score fusion).

Step 7 ("deep learning model such as EfficientNet-B4 / ViT for AI-generation
detection") CANNOT be honestly faked. A real AI-image detector needs weights
trained on large labeled datasets of real-vs-generated images, which are not
available in this environment (no internet access to model hubs, no training
data). Rather than pretend a random/untrained network is "detecting AI
images" — which would just be noise dressed up as a percentage — this module:
  (a) implements a well-documented, non-ML *signal-based* heuristic (frequency
      spectrum regularity + noise-residual statistics) that correlates with
      known GAN/diffusion artifacts, clearly labeled as a heuristic, and
  (b) exposes a clean plug-in point (`run_deep_model_stub`) where a real
      trained EfficientNet-B4 / ViT checkpoint can be dropped in (e.g. via
      timm + your own fine-tuned weights, or a hosted API) to replace the
      heuristic with an actual model score.
"""

import io
import os
import json
import math
import numpy as np
from PIL import Image, ExifTags
import cv2

try:
    import piexif
    HAVE_PIEXIF = True
except ImportError:
    HAVE_PIEXIF = False


def _cv2_imread_safe(path, flags=None):
    """cv2.imread() can silently return None on some Windows setups (path
    encoding quirks, certain OpenCV wheel builds, etc.) even when the file
    is perfectly valid. This reads the bytes ourselves and decodes them,
    which sidesteps that issue, and falls back to a clear error with the
    resolved path if the file is genuinely unreadable/corrupt."""
    if flags is None:
        flags = cv2.IMREAD_COLOR
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"Image file not found: {abspath}")
    with open(abspath, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        try:
            pil_img = Image.open(abspath)
            if flags == cv2.IMREAD_GRAYSCALE:
                pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            else:
                pil_img = pil_img.convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(
                f"Could not read image at {abspath} with OpenCV or Pillow. "
                f"The file may be corrupt or an unsupported format. ({e})"
            )
    return img


# ----------------------------------------------------------------------
# 1. METADATA / EXIF EXTRACTION
# ----------------------------------------------------------------------

KNOWN_EDIT_SOFTWARE_SIGNATURES = [
    "Adobe Photoshop", "Photoshop", "Lightroom", "Adobe Lightroom",
    "GIMP", "Snapseed", "Facetune", "Canva", "Pixelmator", "Affinity Photo",
    "Capture One", "Luminar", "PicsArt", "VSCO", "instagram",
]
KNOWN_AI_GEN_SIGNATURES = [
    "Midjourney", "DALL-E", "DALL·E", "Stable Diffusion", "stability.ai",
    "NovelAI", "Adobe Firefly", "Leonardo.Ai", "Ideogram", "Bing Image Creator",
    "Imagen", "Runway", "Flux", "comfyui", "automatic1111",
]


def extract_metadata(path):
    result = {
        "camera_make": None, "camera_model": None, "lens_model": None,
        "capture_datetime": None, "software": None, "gps": None,
        "exif_present": False, "resolution": None, "color_profile": None,
        "editing_software_detected": None, "ai_generator_detected": None,
        "raw_exif_fields": {},
    }
    try:
        img = Image.open(path)
        result["resolution"] = f"{img.width}x{img.height}"
        result["color_profile"] = img.info.get("icc_profile") and "Embedded ICC profile present" or "None detected"

        exif_data = img.getexif()
        if exif_data and len(exif_data) > 0:
            result["exif_present"] = True
            tags = {}
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                tags[str(tag)] = str(value)
            result["raw_exif_fields"] = tags

            result["camera_make"] = tags.get("Make")
            result["camera_model"] = tags.get("Model")
            result["software"] = tags.get("Software")
            result["capture_datetime"] = tags.get("DateTimeOriginal") or tags.get("DateTime")

            # GPS (IFD)
            try:
                gps_ifd = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                    lat = _convert_gps(gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef"))
                    lon = _convert_gps(gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef"))
                    if lat is not None and lon is not None:
                        result["gps"] = {"lat": lat, "lon": lon}
            except Exception:
                pass

            # Lens model sometimes in EXIF IFD
            try:
                exif_ifd = exif_data.get_ifd(ExifTags.IFD.Exif)
                lens = exif_ifd.get(42036)  # LensModel tag
                if lens:
                    result["lens_model"] = str(lens)
            except Exception:
                pass
        else:
            result["exif_present"] = False

        # Check software / XMP-ish text for editing or AI-gen signatures
        combined_text = " ".join(
            [str(result.get("software") or "")]
            + [f"{k}:{v}" for k, v in result["raw_exif_fields"].items()]
        )
        # Also scan raw file bytes (first/last 64kb) for XMP/AI generator strings
        # (many AI tools & editors embed identifiers even in PNG/WEBP where
        # classic EXIF is absent).
        try:
            with open(path, "rb") as f:
                head = f.read(65536)
                f.seek(max(0, os.path.getsize(path) - 65536))
                tail = f.read(65536)
            blob = (head + tail).decode("latin-1", errors="ignore")
        except Exception:
            blob = ""
        combined_text += " " + blob

        for sig in KNOWN_EDIT_SOFTWARE_SIGNATURES:
            if sig.lower() in combined_text.lower():
                result["editing_software_detected"] = sig
                break
        for sig in KNOWN_AI_GEN_SIGNATURES:
            if sig.lower() in combined_text.lower():
                result["ai_generator_detected"] = sig
                break

    except Exception as e:
        result["error"] = str(e)

    return result


def _convert_gps(dms, ref):
    if not dms or not ref:
        return None
    try:
        d, m, s = [float(x) for x in dms]
        val = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            val = -val
        return round(val, 6)
    except Exception:
        return None


# ----------------------------------------------------------------------
# 2. ERROR LEVEL ANALYSIS (ELA)
# ----------------------------------------------------------------------

def error_level_analysis(path, quality=90, scale=15):
    """Resave the image at a known JPEG quality and diff against the
    original. Regions that were edited/spliced after the last save tend
    to have a different, elevated error level than untouched regions."""
    orig = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    orig.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    orig_np = np.asarray(orig).astype(np.int16)
    resaved_np = np.asarray(resaved).astype(np.int16)
    diff = np.abs(orig_np - resaved_np).astype(np.uint8)
    diff_gray = diff.max(axis=2)

    ela_amplified = np.clip(diff_gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # Tile into blocks, compute mean error per block -> find outlier blocks
    h, w = ela_amplified.shape
    tile = 32
    block_means = []
    coords = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            block = ela_amplified[y:y + tile, x:x + tile]
            block_means.append(block.mean())
            coords.append((x, y))
    block_means = np.array(block_means) if block_means else np.array([0.0])

    mean_err = float(block_means.mean())
    std_err = float(block_means.std())
    max_err = float(block_means.max())
    # outlier blocks = blocks with error > mean + 2.5*std (likely re-edited regions)
    if std_err > 1e-6:
        outlier_ratio = float((block_means > (mean_err + 2.5 * std_err)).mean())
    else:
        outlier_ratio = 0.0

    return {
        "ela_map": ela_amplified,       # for heatmap fusion
        "mean_error": mean_err,
        "std_error": std_err,
        "max_error": max_err,
        "outlier_block_ratio": outlier_ratio,
    }


# ----------------------------------------------------------------------
# 3. NOISE / SENSOR PATTERN ANALYSIS
# ----------------------------------------------------------------------

def noise_analysis(path):
    """Estimate sensor noise residual and check spatial consistency.
    Real camera photos show a fairly consistent grain pattern driven by
    sensor + ISO. Spliced regions or heavily denoised/AI content show
    abrupt changes in local noise energy across the image."""
    img = _cv2_imread_safe(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image with OpenCV")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # High-pass residual: original - denoised
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - denoised

    # Local noise energy map via block std
    tile = 24
    h, w = residual.shape
    local_std = []
    std_map = np.zeros_like(residual)
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            block = residual[y:y + tile, x:x + tile]
            s = float(block.std())
            local_std.append(s)
            std_map[y:y + tile, x:x + tile] = s
    local_std = np.array(local_std) if local_std else np.array([0.0])

    global_noise_energy = float(residual.std())
    noise_consistency_cv = float(local_std.std() / (local_std.mean() + 1e-6))  # coeff. of variation
    # high CV => noise energy varies a lot across regions => possible splicing
    # very low global noise energy => possible AI-smoothness or heavy denoise/beautify

    return {
        "noise_map": std_map,
        "global_noise_energy": global_noise_energy,
        "noise_consistency_cv": noise_consistency_cv,
    }


# ----------------------------------------------------------------------
# 4. JPEG COMPRESSION ARTIFACT / BLOCKINESS DETECTION
# ----------------------------------------------------------------------

def jpeg_artifact_analysis(path):
    """Measures 8x8 blocking-grid discontinuity strength, a signature of
    JPEG compression. Its absence in a file claiming to be a photo, or its
    presence in unexpected alignment, is diagnostic. Also flags likely
    double-compression (common after re-saving an edited JPEG)."""
    img = _cv2_imread_safe(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"blockiness_score": 0.0, "likely_recompressed": False}
    img = img.astype(np.float32)
    h, w = img.shape

    # Sum absolute difference across 8-pixel grid boundaries vs. non-boundary
    boundary_diffs = []
    nonboundary_diffs = []
    for x in range(1, w - 1):
        col_diff = np.abs(img[:, x] - img[:, x - 1]).mean()
        if x % 8 == 0:
            boundary_diffs.append(col_diff)
        else:
            nonboundary_diffs.append(col_diff)

    b_mean = float(np.mean(boundary_diffs)) if boundary_diffs else 0.0
    nb_mean = float(np.mean(nonboundary_diffs)) if nonboundary_diffs else 1e-6
    blockiness_score = float(b_mean / (nb_mean + 1e-6))  # >1 => visible 8x8 grid

    ext = os.path.splitext(path)[1].lower()
    likely_recompressed = blockiness_score > 1.15 and ext in (".jpg", ".jpeg")

    return {
        "blockiness_score": round(blockiness_score, 3),
        "likely_recompressed": bool(likely_recompressed),
    }


# ----------------------------------------------------------------------
# 5. COLOR HISTOGRAM / LUT / GRADING ANALYSIS
# ----------------------------------------------------------------------

def color_histogram_analysis(path):
    img = _cv2_imread_safe(path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    findings = {}
    comb_counts = []
    clip_ratios = []
    for name, ch in [("blue", b), ("green", g), ("red", r)]:
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        # "Comb" pattern: alternating near-zero bins is a classic sign of
        # levels/curves or LUT remapping stretching the histogram.
        zero_ish = int(np.sum(hist < (hist.mean() * 0.05)))
        comb_counts.append(zero_ish)
        clip_low = float(hist[0] / hist.sum())
        clip_high = float(hist[-1] / hist.sum())
        clip_ratios.append(clip_low + clip_high)
        findings[f"{name}_channel_clip_ratio"] = round(clip_low + clip_high, 4)

    avg_comb = float(np.mean(comb_counts))
    comb_pattern_detected = avg_comb > 40  # many empty/sparse bins => stretched histogram

    saturation_mean = float(s_ch.mean())
    saturation_std = float(s_ch.std())
    value_mean = float(v_ch.mean())
    contrast_std = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    # Heuristic thresholds for "excessive" grading
    excessive_saturation = saturation_mean > 150 or saturation_std > 75
    excessive_clipping = float(np.mean(clip_ratios)) > 0.02
    high_contrast_push = contrast_std > 70

    return {
        "saturation_mean": round(saturation_mean, 2),
        "saturation_std": round(saturation_std, 2),
        "brightness_mean": round(value_mean, 2),
        "contrast_std": round(contrast_std, 2),
        "avg_clip_ratio": round(float(np.mean(clip_ratios)), 4),
        "comb_pattern_score": round(avg_comb, 1),
        "comb_pattern_detected": bool(comb_pattern_detected),
        "excessive_saturation": bool(excessive_saturation),
        "excessive_clipping": bool(excessive_clipping),
        "high_contrast_push": bool(high_contrast_push),
        "per_channel": findings,
    }


# ----------------------------------------------------------------------
# 6. LIGHTING / SHADOW / EDGE CONSISTENCY (heuristic)
# ----------------------------------------------------------------------

def lighting_edge_consistency(path):
    """Rough heuristic: estimate a local 'illumination gradient direction'
    per grid cell from image luminance gradients, then measure how much
    that direction disagrees across the image. Real single-light-source
    photos tend to have a fairly coherent gradient direction; composites
    with mismatched lighting show higher directional variance. This is a
    coarse proxy, not true 3D shadow/reflection reasoning."""
    img = _cv2_imread_safe(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)

    h, w = img.shape
    tile = max(32, min(h, w) // 8)
    directions = []
    edge_densities = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            bx = gx[y:y + tile, x:x + tile]
            by = gy[y:y + tile, x:x + tile]
            mag = np.sqrt(bx ** 2 + by ** 2)
            if mag.mean() < 3:  # flat region, skip (no reliable gradient)
                continue
            # dominant direction weighted by magnitude
            ang = math.atan2(float(by.mean()), float(bx.mean()))
            directions.append(ang)
            edge_densities.append(float((mag > 30).mean()))

    if len(directions) < 4:
        return {"lighting_variance_score": 0.0, "edge_density_mean": 0.0, "insufficient_texture": True}

    directions = np.array(directions)
    # circular variance
    sin_mean = np.mean(np.sin(directions))
    cos_mean = np.mean(np.cos(directions))
    r = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
    circular_variance = 1 - r  # 0 = perfectly consistent, 1 = totally random

    return {
        "lighting_variance_score": round(float(circular_variance), 3),
        "edge_density_mean": round(float(np.mean(edge_densities)), 3),
        "insufficient_texture": False,
    }


# ----------------------------------------------------------------------
# 7. AI-GENERATION HEURISTIC (frequency + noise signal-based proxy)
#    See module docstring: this is NOT a trained EfficientNet/ViT model.
# ----------------------------------------------------------------------

def ai_generation_heuristic(path):
    img = _cv2_imread_safe(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    h, w = img.shape
    side = min(h, w)
    side = side - (side % 2)
    crop = img[:side, :side] if side > 0 else img

    # FFT magnitude spectrum, radially averaged
    f = np.fft.fftshift(np.fft.fft2(crop))
    mag = np.log(np.abs(f) + 1)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    max_r = min(cy, cx)
    radial_profile = []
    yy, xx = np.indices(mag.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    for radius in range(1, max_r):
        ring = mag[r == radius]
        if ring.size:
            radial_profile.append(float(ring.mean()))
    radial_profile = np.array(radial_profile) if radial_profile else np.array([0.0])

    # 1) High-frequency energy ratio: natural photos retain sensor-noise
    #    energy out to the Nyquist edge; many GAN/diffusion pipelines
    #    (esp. after upsampling/decoding) show a steeper falloff or
    #    unnatural periodic bumps.
    n = len(radial_profile)
    low = radial_profile[: n // 4].mean() if n >= 4 else radial_profile.mean()
    high = radial_profile[3 * n // 4:].mean() if n >= 4 else radial_profile.mean()
    hf_ratio = float(high / (low + 1e-6))

    # 2) Periodicity / spikiness in the radial profile (upsampling checkerboard
    #    artifacts from transposed-conv/decoder architectures show up as
    #    regular bumps in the radial spectrum).
    if n > 8:
        detrended = radial_profile - np.convolve(radial_profile, np.ones(9) / 9, mode="same")
        spikiness = float(np.std(detrended) / (np.mean(np.abs(radial_profile)) + 1e-6))
    else:
        spikiness = 0.0

    # 3) Noise residual "too clean" signal (reuse a cheap residual calc)
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    residual_energy = float((img - denoised).std())

    # Combine into a 0-1 heuristic "AI-likelihood" signal.
    # These weights/thresholds are illustrative, tuned by hand on general
    # intuition about known artifacts, NOT validated against a labeled
    # benchmark. Treat as a coarse signal, not ground truth.
    score = 0.0
    score += 0.35 * np.clip((0.55 - hf_ratio) / 0.45, 0, 1)       # low HF energy -> more AI-like
    score += 0.35 * np.clip((spikiness - 0.15) / 0.5, 0, 1)        # periodic spectral bumps -> more AI-like
    score += 0.30 * np.clip((6.0 - residual_energy) / 6.0, 0, 1)   # very low sensor noise -> more AI-like
    score = float(np.clip(score, 0, 1))

    return {
        "hf_energy_ratio": round(hf_ratio, 4),
        "spectral_spikiness": round(spikiness, 4),
        "residual_noise_energy": round(residual_energy, 4),
        "ai_heuristic_score": round(score, 4),  # 0 = looks natural, 1 = looks synthetic
    }


def run_deep_model_stub(path):
    """
    PLUG-IN POINT for a real deep-learning AI-image detector.

    To make this production-grade, replace this function body with a call
    to an actual trained classifier, e.g.:

        import timm, torch
        model = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=1)
        model.load_state_dict(torch.load('your_finetuned_ai_detector.pt'))
        model.eval()
        # preprocess `path` -> tensor -> model(tensor) -> sigmoid -> probability

    or call a hosted detection API. This stub deliberately returns None so
    the rest of the pipeline clearly falls back to the disclosed heuristic
    above rather than silently faking a model output.
    """
    return None


# ----------------------------------------------------------------------
# 8. HEATMAP GENERATION
# ----------------------------------------------------------------------

def build_heatmap(path, ela_map, noise_std_map, out_path):
    orig = _cv2_imread_safe(path, cv2.IMREAD_COLOR)
    h, w = orig.shape[:2]

    ela_resized = cv2.resize(ela_map.astype(np.float32), (w, h))
    noise_resized = cv2.resize(noise_std_map.astype(np.float32), (w, h))

    def norm(m):
        mn, mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-6)

    combined = 0.6 * norm(ela_resized) + 0.4 * norm(noise_resized)
    combined_u8 = np.clip(combined * 255, 0, 255).astype(np.uint8)
    combined_u8 = cv2.GaussianBlur(combined_u8, (9, 9), 0)

    heat_color = cv2.applyColorMap(combined_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heat_color, 0.4, 0)

    # Bounding boxes around top suspicious regions
    thresh_val = np.percentile(combined_u8, 97)
    _, mask = cv2.threshold(combined_u8, thresh_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = (w * h) * 0.001
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh))
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:8]

    cv2.imwrite(out_path, overlay)
    return {"heatmap_path": out_path, "suspicious_regions": boxes, "num_regions": len(boxes)}
