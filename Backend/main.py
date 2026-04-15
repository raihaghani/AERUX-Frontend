import asyncio, json, os, shutil, time, uuid, zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from constants import LOCATION_LABELS
from scripts.inference_fusion import load_fusion_model
from scripts.infer_single_overlay import (
    build_location_prior_heatmap,
    colorize_and_overlay,
)
from scripts.preprocess_dataset import (
    load_dicom_series,
    extract_2_5d_slices,
    resize_volume,
    normalize_volume,
)
from src.datasets.preprocessing import MRIPreprocessor, CTAPreprocessor

from dotenv import load_dotenv

# Load local .env first
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR",  "./uploads"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR",  "./inference_outputs"))
FUSION_PATH = os.getenv("FUSION_MODEL_PATH")
MAX_BYTES   = int(os.getenv("MAX_UPLOAD_MB", 200)) * 1024 * 1024
TTL_HOURS   = int(os.getenv("RESULT_TTL_HOURS", 24))
TARGET_SIZE = 256

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".dcm", ".npy", ".png", ".jpg", ".jpeg", ".nii", ".zip"}
ALLOWED_MOD = {"CTA", "MRA", "MRI"}

device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model    = None
MODEL_LOADED_AT: str | None = None

# ── Preprocessors (match training config exactly) ─────────────────────────────
PREPROCESSORS: dict[str, MRIPreprocessor | CTAPreprocessor] = {
    "MRI": MRIPreprocessor(
        apply_n4=True, apply_vesselness=True, apply_otsu=False,
        vesselness_scale_range=(1, 8), vesselness_scale_step=2,
        n4_max_iterations=[50, 50, 50, 50],
    ),
    "MRA": MRIPreprocessor(
        apply_n4=True, apply_vesselness=True, apply_otsu=False,
        vesselness_scale_range=(1, 8), vesselness_scale_step=2,
        n4_max_iterations=[50, 50, 50, 50],
    ),
    "CTA": CTAPreprocessor(
        apply_hu_windowing=True, apply_clahe=False, apply_vesselness=True,
        window_center=40.0, window_width=80.0,
        vesselness_scale_range=(1, 8), vesselness_scale_step=2,
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fusion_model, MODEL_LOADED_AT
    if not FUSION_PATH or not os.path.exists(FUSION_PATH):
        print(f"[ERROR] FUSION_MODEL_PATH is invalid or file missing: {FUSION_PATH}")
    else:
        print(f"[startup] Loading fusion model: {FUSION_PATH}")
        fusion_model = load_fusion_model(FUSION_PATH, device)
        fusion_model.eval()
        MODEL_LOADED_AT = datetime.utcnow().isoformat() + "Z"
        print("[startup] Model ready.")
    yield

app = FastAPI(title="AERUX ML API", version="1.0.0", lifespan=lifespan)

# CORS: allow only the Next.js server (now allowing * for ngrok external access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)


def _process_dicom_series(series_dir: Path, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the full training preprocessing pipeline on a directory of DICOM files.

    Pipeline:  load_dicom_series → extract_2_5d (7 slices) → resize 256×256
               → per-slice modality preprocessing → normalize → (7, 256, 256)

    Matches preprocess_dataset.py → preprocess_series() exactly.
    """
    vol_3d, _meta, _sop_map = load_dicom_series(str(series_dir))
    if vol_3d is None:
        raise ValueError("Could not load any DICOM slices from the uploaded files")

    vol_25d = extract_2_5d_slices(vol_3d, num_slices=7)          # (H, W, 7)
    vol_25d = resize_volume(vol_25d, (TARGET_SIZE, TARGET_SIZE))  # (256, 256, 7)

    # Display image from raw center slice BEFORE preprocessing
    raw_center = vol_25d[:, :, 3]
    rc_min, rc_max = float(raw_center.min()), float(raw_center.max())
    if rc_max > rc_min:
        display = ((raw_center - rc_min) / (rc_max - rc_min) * 255.0).astype(np.uint8)
    else:
        display = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    prep = PREPROCESSORS.get(modality)
    if prep is not None:
        vol_25d = np.stack(
            [prep.preprocess(vol_25d[:, :, i])["preprocessed"] for i in range(7)],
            axis=-1,
        )

    vol_25d = normalize_volume(vol_25d)                            # [0, 1]
    vol_7ch = np.transpose(vol_25d, (2, 0, 1)).astype(np.float32) # (7, 256, 256)

    return vol_7ch, display


def _load_npy_as_7ch(upload_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an already-preprocessed .npy from the training pipeline.

    Returns all 7 channels untouched — NO re-preprocessing.
    This matches exactly what infer_single_overlay.py does.
    """
    arr = np.load(upload_path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D npy array, got shape {arr.shape}")

    if arr.shape[0] == 7:
        vol = arr                               # (7, H, W)
    elif arr.shape[-1] == 7:
        vol = np.transpose(arr, (2, 0, 1))     # (H, W, 7) → (7, H, W)
    else:
        raise ValueError(f"Expected 7 channels, got shape {arr.shape}")

    # Resize each channel if needed
    h, w = vol.shape[1], vol.shape[2]
    if (h, w) != (TARGET_SIZE, TARGET_SIZE):
        resized = [
            cv2.resize(vol[c], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            for c in range(7)
        ]
        vol = np.stack(resized, axis=0).astype(np.float32)

    # Display image from center slice
    center = vol[3]
    c_min, c_max = float(center.min()), float(center.max())
    if c_max > c_min:
        display = ((center - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
    else:
        display = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    return vol, display


def _load_raw_image(upload_path: Path) -> np.ndarray:
    """Load a raw (non-.npy) upload as a 2D float32 grayscale array (H, W)."""
    suffix = upload_path.suffix.lower()
    filename_lower = upload_path.name.lower()

    if suffix in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(str(upload_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {upload_path}")
        return img.astype(np.float32)

    if suffix == ".dcm":
        import pydicom
        dcm = pydicom.dcmread(str(upload_path))
        img = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        return img

    if suffix == ".nii" or filename_lower.endswith(".nii.gz"):
        import SimpleITK as sitk
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(upload_path))).astype(np.float32)
        return vol[vol.shape[0] // 2]

    raise HTTPException(400, f"Unsupported format '{suffix}'.")


def preprocess_upload(
    upload_path: Path, modality: str | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Prepare an uploaded file for inference.

    *modality* may be ``None`` (auto-detect).  For DICOM files the modality
    is read from the file header.  For .npy / generic images it is left as
    ``None`` and the caller should fall back to ``_infer_best_modality``.

    Returns:
        vol_7ch          — float32 (7, 256, 256), values in [0, 1]
        display_gray     — uint8   (256, 256),     values in [0, 255]
        detected_modality — the modality that was used (or None when unknown)
    """
    suffix = upload_path.suffix.lower()

    # ── .npy: already preprocessed — pass through ─────────────────────────
    if suffix == ".npy":
        vol, display = _load_npy_as_7ch(upload_path)
        if modality is None:
            modality = _detect_modality_from_filename(upload_path)
        return vol, display, modality

    # ── .zip: DICOM series archive — full training pipeline ───────────────
    if suffix == ".zip":
        extract_dir = upload_path.parent / "dicom_series"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(upload_path, "r") as zf:
            zf.extractall(extract_dir)

        dcm_dir = _find_dcm_directory(extract_dir)
        if dcm_dir is None:
            raise HTTPException(400, "No .dcm files found inside the uploaded zip")

        if modality is None:
            first = _find_first_dcm(dcm_dir)
            modality = _detect_modality_from_dicom(first) if first else "CTA"
            print(f"[INFO] Auto-detected modality from DICOM series: {modality}")

        vol, display = _process_dicom_series(dcm_dir, modality)
        return vol, display, modality

    # ── Single 2D image (.dcm / .png / .jpg / .nii) ──────────────────────
    if suffix == ".dcm" and modality is None:
        modality = _detect_modality_from_dicom(upload_path)
        print(f"[INFO] Auto-detected modality from DICOM header: {modality}")
    elif modality is None:
        modality = _detect_modality_from_filename(upload_path)

    raw = _load_raw_image(upload_path)
    raw = cv2.resize(raw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    raw_min, raw_max = float(raw.min()), float(raw.max())
    if raw_max > raw_min:
        display_gray = ((raw - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    else:
        display_gray = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    if modality is not None:
        prep = PREPROCESSORS.get(modality)
        if prep is not None:
            processed = prep.preprocess(raw)["preprocessed"]
        else:
            processed = raw
    else:
        processed = raw

    p_min, p_max = float(processed.min()), float(processed.max())
    if p_max - p_min > 1e-6:
        normed = ((processed - p_min) / (p_max - p_min)).astype(np.float32)
    else:
        normed = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

    vol_7ch = np.stack([normed] * 7, axis=0)

    if suffix == ".dcm":
        print(f"[INFO] Single DICOM uploaded — 7 identical channels. "
              f"For accurate 2.5D inference, upload the full series as .zip.")

    return vol_7ch, display_gray, modality


def _find_dcm_directory(root: Path) -> Path | None:
    """Walk a directory tree and return the first folder containing .dcm files."""
    for dirpath, _dirs, files in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in files):
            return Path(dirpath)
    return None


def _find_first_dcm(directory: Path) -> Path | None:
    """Return the first .dcm file found under *directory*."""
    for dirpath, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".dcm"):
                return Path(dirpath) / f
    return None


def _detect_modality_from_dicom(dcm_path: Path) -> str:
    """Read DICOM header metadata and map to CTA / MRA / MRI."""
    try:
        import pydicom
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)

        base = getattr(dcm, "Modality", "").upper().strip()
        desc = getattr(dcm, "SeriesDescription", "").upper()
        proto = getattr(dcm, "ProtocolName", "").upper()
        combined = f"{base} {desc} {proto}"

        if base == "CT" or "CTA" in combined:
            return "CTA"
        if "MRA" in combined or "TOF" in combined or "ANGIO" in combined:
            return "MRA"
        if "T1" in combined or "T2" in combined:
            return "MRI"
        if base == "MR":
            return "MRA"

        return "CTA"
    except Exception as exc:
        print(f"[WARN] Could not read DICOM metadata for modality detection: {exc}")
        return "CTA"


def _detect_modality_from_filename(path: Path) -> str | None:
    """Try to extract CTA / MRA / MRI from the filename (case-insensitive)."""
    name = path.stem.upper()
    if "CTA" in name:
        return "CTA"
    if "MRA" in name:
        return "MRA"
    if "MRI" in name or "T1" in name or "T2" in name:
        return "MRI"
    return None


def _infer_best_modality(vol_7ch: np.ndarray) -> str:
    """Run all 3 modality backbones and return the one with highest confidence."""
    best_mod = "CTA"
    best_conf = 0.0
    for mod in ALLOWED_MOD:
        det_probs, _ = _quick_inference(vol_7ch, mod)
        conf = float(max(det_probs[0], det_probs[1]))
        if conf > best_conf:
            best_conf = conf
            best_mod = mod
    print(f"[INFO] Auto-detected modality via inference: {best_mod} (confidence={best_conf:.3f})")
    return best_mod


# ── Series-scan helpers ────────────────────────────────────────────────────────

def _preprocess_all_slices(
    vol_3d: np.ndarray, modality: str, fast: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize every slice to 256×256 and apply modality preprocessing.

    Returns:
        preprocessed — (D, 256, 256) preprocessed (NOT yet window-normalized)
        raw_resized  — (D, 256, 256) raw resized (for readable overlay display)

    In *fast* mode, N4 bias-field correction is skipped for MRI/MRA.
    """
    D = vol_3d.shape[0]

    if fast and modality in ("MRI", "MRA"):
        prep: MRIPreprocessor | CTAPreprocessor | None = MRIPreprocessor(
            apply_n4=False, apply_vesselness=True, apply_otsu=False,
            vesselness_scale_range=(1, 8), vesselness_scale_step=2,
        )
    else:
        prep = PREPROCESSORS.get(modality)

    raw_resized  = np.empty((D, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    preprocessed = np.empty((D, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for i in range(D):
        sl = cv2.resize(
            vol_3d[i].astype(np.float32),
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        raw_resized[i] = sl
        if prep is not None:
            sl = prep.preprocess(sl)["preprocessed"].astype(np.float32)
        preprocessed[i] = sl
    return preprocessed, raw_resized


def _build_window(
    preprocessed: np.ndarray, raw_resized: np.ndarray, center: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalized 7-channel input + display image for a given center."""
    D = preprocessed.shape[0]
    indices = [max(0, min(D - 1, center + off)) for off in range(-3, 4)]
    window = preprocessed[indices]                                        # (7, 256, 256)

    window_hwc = np.transpose(window, (1, 2, 0))                         # (256, 256, 7)
    window_hwc = normalize_volume(window_hwc)
    vol_7ch = np.transpose(window_hwc, (2, 0, 1)).astype(np.float32)     # (7, 256, 256)

    # Display from raw center slice (readable brain image, not vesselness)
    raw_center = raw_resized[indices[3]]
    mn, mx = float(raw_center.min()), float(raw_center.max())
    if mx > mn:
        display = ((raw_center - mn) / (mx - mn) * 255.0).astype(np.uint8)
    else:
        display = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    return vol_7ch, display


def _quick_inference(vol_7ch: np.ndarray, modality: str):
    """Forward pass returning only probability arrays (no overlay generation)."""
    x = torch.from_numpy(vol_7ch).unsqueeze(0).to(device)
    inputs           = {"CTA": None, "MRA": None, "MRI": None}
    modality_indices = {"CTA": [],   "MRA": [],   "MRI": []}
    inputs[modality]           = x
    modality_indices[modality] = [0]

    with torch.no_grad():
        det_logits, loc_logits, _ = fusion_model(inputs, modality_indices)

    det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]
    loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]
    return det_probs, loc_probs


def _series_scan_sync(
    preprocessed: np.ndarray,
    raw_resized: np.ndarray,
    modality: str,
    step: int,
    sigma: float,
    alpha: float,
    threshold: float,
    use_detection_weight: bool,
    top_k: int,
    out_dir: Path,
    result_id: str,
):
    """Two-pass sliding-window scan (blocking — meant for run_in_executor).

    Pass 1: quick probability scan over every window.
    Pass 2: full inference + overlay generation for the top-K windows.
    """
    D = preprocessed.shape[0]

    # ── Pass 1 ─────────────────────────────────────────────────────────────
    window_results: list[dict] = []
    for center in range(0, D, step):
        vol_7ch, _ = _build_window(preprocessed, raw_resized, center)
        det_probs, _ = _quick_inference(vol_7ch, modality)
        window_results.append({
            "center_slice": int(center),
            "aneurysm_prob": round(float(det_probs[1]), 4),
        })

    ranked = sorted(window_results, key=lambda w: w["aneurysm_prob"], reverse=True)
    top_centers = [w["center_slice"] for w in ranked[:top_k]]

    # ── Pass 2 ─────────────────────────────────────────────────────────────
    top_details: list[dict] = []
    for center in top_centers:
        vol_7ch, display = _build_window(preprocessed, raw_resized, center)
        (
            det_probs, loc_probs, det_pred, loc_weights,
            heat_uint8, binary_mask, heat_color, overlay,
        ) = _run_inference_sync(
            vol_7ch, display, modality,
            sigma, alpha, threshold, use_detection_weight,
        )

        prefix = f"window_{center}"
        cv2.imwrite(str(out_dir / f"{prefix}_overlay.png"),        overlay)
        cv2.imwrite(str(out_dir / f"{prefix}_heatmap.png"),        heat_color)
        cv2.imwrite(str(out_dir / f"{prefix}_input_gray.png"),     display)
        cv2.imwrite(str(out_dir / f"{prefix}_highlight_mask.png"), binary_mask)

        top3 = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)[:3]

        top_details.append({
            "center_slice": int(center),
            "detection_prediction": det_pred,
            "detection_probabilities": {
                "no_aneurysm": round(float(det_probs[0]), 4),
                "aneurysm":    round(float(det_probs[1]), 4),
            },
            "top_3_locations": [
                {"label": label, "score": round(float(score), 4)}
                for label, score in top3
            ],
            "all_location_probabilities": {
                LOCATION_LABELS[i]: round(float(loc_probs[i]), 4) for i in range(13)
            },
            "image_urls": {
                "overlay":        f"/results/{result_id}/{prefix}_overlay.png",
                "heatmap":        f"/results/{result_id}/{prefix}_heatmap.png",
                "input_gray":     f"/results/{result_id}/{prefix}_input_gray.png",
                "highlight_mask": f"/results/{result_id}/{prefix}_highlight_mask.png",
            },
        })

    return window_results, top_details


def _run_inference_sync(
    vol_7ch: np.ndarray,
    display_gray: np.ndarray,
    modality: str,
    sigma: float,
    alpha: float,
    threshold: float,
    use_detection_weight: bool,
):
    """Blocking inference — runs in thread pool, never on the event loop."""
    x = torch.from_numpy(vol_7ch).unsqueeze(0).to(device)   # → (1, 7, 256, 256)

    inputs           = {"CTA": None, "MRA": None, "MRI": None}
    modality_indices = {"CTA": [],   "MRA": [],   "MRI": []}
    inputs[modality]           = x
    modality_indices[modality] = [0]

    with torch.no_grad():
        det_logits, loc_logits, _ = fusion_model(inputs, modality_indices)

    det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]
    loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]
    det_pred  = int(np.argmax(det_probs))
    det_prob  = float(det_probs[1])

    heat, loc_weights = build_location_prior_heatmap(
        det_prob, loc_probs, TARGET_SIZE, TARGET_SIZE, sigma, use_detection_weight
    )
    heat_uint8, binary_mask, heat_color, overlay = colorize_and_overlay(
        display_gray, heat, alpha, threshold
    )

    return (
        det_probs, loc_probs, det_pred, loc_weights,
        heat_uint8, binary_mask, heat_color, overlay
    )


def _cleanup_old_results():
    """Delete result dirs older than TTL_HOURS. Called as a background task."""
    cutoff = datetime.utcnow() - timedelta(hours=TTL_HOURS)
    for base in (OUTPUT_DIR, UPLOAD_DIR):
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir():
                mtime = datetime.utcfromtimestamp(d.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)


@app.get("/")
def root():
    return {"message": "AERUX ML API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_loaded":      fusion_model is not None,
        "model_loaded_at":   MODEL_LOADED_AT,
        "device":            str(device),
        "gpu_name":          (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and device.type == "cuda" else "cpu"
        ),
        "fusion_model_path": FUSION_PATH,
        "api_version":       "1.0.0",
    }


@app.post("/predict")
async def predict(
    background_tasks:     BackgroundTasks,
    file:                 UploadFile = File(...),
    modality:             str        = Form("auto"),
    sigma:                float      = Form(18.0),
    alpha:                float      = Form(0.45),
    threshold:            float      = Form(0.45),
    use_detection_weight: bool       = Form(True),
):
    if fusion_model is None:
        raise HTTPException(503, "Model not yet loaded or missing FUSION_MODEL_PATH configuration")

    effective_mod: str | None = modality if modality in ALLOWED_MOD else None

    # ── Validate file extension ────────────────────────────────────────────
    filename_lower = (file.filename or "").lower()
    ext = Path(filename_lower).suffix

    is_nifti_gz = filename_lower.endswith(".nii.gz")
    if not is_nifti_gz and ext not in ALLOWED_EXT:
        raise HTTPException(
            400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXT} or .nii.gz"
        )

    # ── Validate file size ─────────────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES // (1024 * 1024)} MB limit")

    # ── Save upload ────────────────────────────────────────────────────────
    result_id   = str(uuid.uuid4())[:8].lower()
    upload_dir  = UPLOAD_DIR / result_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    real_filename = file.filename if file.filename else f"upload{ext}"
    upload_path = upload_dir / real_filename
    upload_path.write_bytes(file_bytes)

    # ── Preprocess ─────────────────────────────────────────────────────────
    try:
        vol_7ch, display_gray, detected_mod = preprocess_upload(upload_path, effective_mod)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # If modality couldn't be determined from the file, try all 3 backbones
    if detected_mod is None:
        detected_mod = _infer_best_modality(vol_7ch)

    # ── Inference (non-blocking — runs in thread pool) ─────────────────────
    t0 = time.time()
    loop = asyncio.get_running_loop()
    try:
        (
            det_probs, loc_probs, det_pred, loc_weights,
            heat_uint8, binary_mask, heat_color, overlay
        ) = await loop.run_in_executor(
            None,
            _run_inference_sync,
            vol_7ch, display_gray, detected_mod, sigma, alpha, threshold, use_detection_weight,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    elapsed_ms = int((time.time() - t0) * 1000)

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = OUTPUT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "overlay.png"),        overlay)
    cv2.imwrite(str(out_dir / "heatmap.png"),         heat_color)
    cv2.imwrite(str(out_dir / "input_gray.png"),      display_gray)
    cv2.imwrite(str(out_dir / "highlight_mask.png"),  binary_mask)

    top3 = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)[:3]

    result = {
        "result_id":               result_id,
        "detected_modality":       detected_mod,
        "detection_prediction":    det_pred,
        "detection_probabilities": {
            "no_aneurysm": float(det_probs[0]),
            "aneurysm":    float(det_probs[1]),
        },
        "top_3_locations": [
            {"label": label, "score": float(score)} for label, score in top3
        ],
        "all_location_probabilities": {
            LOCATION_LABELS[i]: float(loc_probs[i]) for i in range(13)
        },
        "image_urls": {
            "overlay":        f"/results/{result_id}/overlay.png",
            "heatmap":        f"/results/{result_id}/heatmap.png",
            "input_gray":     f"/results/{result_id}/input_gray.png",
            "highlight_mask": f"/results/{result_id}/highlight_mask.png",
        },
        "processing_time_ms": elapsed_ms,
    }

    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2))
    background_tasks.add_task(_cleanup_old_results)

    return result


# ── Series scan endpoint ───────────────────────────────────────────────────────

@app.post("/predict_series")
async def predict_series(
    background_tasks:     BackgroundTasks,
    file:                 UploadFile = File(...),
    modality:             str        = Form("auto"),
    step:                 int        = Form(3),
    sigma:                float      = Form(18.0),
    alpha:                float      = Form(0.45),
    threshold:            float      = Form(0.45),
    use_detection_weight: bool       = Form(True),
    fast_mode:            bool       = Form(False),
    top_k:                int        = Form(5),
):
    """Sliding-window scan over a full DICOM series (uploaded as .zip).

    Preprocesses every slice once, then slides a 7-slice window across the
    volume at the given *step* interval, running the model on each window.
    Full overlays are generated only for the *top_k* most suspicious windows.
    """
    if fusion_model is None:
        raise HTTPException(503, "Model not yet loaded or missing FUSION_MODEL_PATH configuration")

    effective_mod: str | None = modality if modality in ALLOWED_MOD else None

    ext = Path((file.filename or "").lower()).suffix
    if ext != ".zip":
        raise HTTPException(400, "Series scan requires a .zip archive of DICOM images")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES // (1024 * 1024)} MB limit")

    result_id  = str(uuid.uuid4())[:8].lower()
    upload_dir = UPLOAD_DIR / result_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / (file.filename or "upload.zip")
    upload_path.write_bytes(file_bytes)

    # ── Extract zip ────────────────────────────────────────────────────────
    extract_dir = upload_dir / "dicom_series"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(upload_path, "r") as zf:
        zf.extractall(extract_dir)

    dcm_dir = _find_dcm_directory(extract_dir)
    if dcm_dir is None:
        raise HTTPException(400, "No .dcm files found inside the uploaded zip")

    # Auto-detect modality from the first DICOM if not provided
    if effective_mod is None:
        first = _find_first_dcm(dcm_dir)
        effective_mod = _detect_modality_from_dicom(first) if first else "CTA"
        print(f"[INFO] Auto-detected modality from DICOM series: {effective_mod}")

    # ── Load 3-D volume ────────────────────────────────────────────────────
    try:
        vol_3d, _meta, _sop_map = load_dicom_series(str(dcm_dir))
        if vol_3d is None:
            raise ValueError("Could not load DICOM volume")
    except Exception as e:
        raise HTTPException(500, f"DICOM loading failed: {e}")

    # ── Preprocess all slices (expensive, runs in thread pool) ─────────────
    t0   = time.time()
    loop = asyncio.get_running_loop()
    try:
        preprocessed, raw_resized = await loop.run_in_executor(
            None, _preprocess_all_slices, vol_3d, effective_mod, fast_mode,
        )
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # ── Sliding-window scan (thread pool) ──────────────────────────────────
    out_dir = OUTPUT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        slice_results, top_results = await loop.run_in_executor(
            None, _series_scan_sync,
            preprocessed, raw_resized, effective_mod, step, sigma, alpha, threshold,
            use_detection_weight, top_k, out_dir, result_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Series scan failed: {e}")

    elapsed_ms = int((time.time() - t0) * 1000)

    probs     = [r["aneurysm_prob"] for r in slice_results]
    max_prob  = max(probs) if probs else 0.0
    flagged   = sum(1 for p in probs if p >= 0.5)

    result = {
        "result_id":          result_id,
        "mode":               "series",
        "detected_modality":  effective_mod,
        "total_slices":       int(vol_3d.shape[0]),
        "total_windows":      len(slice_results),
        "step":               step,
        "max_aneurysm_prob":  round(max_prob, 4),
        "flagged_windows":    flagged,
        "slice_results":      slice_results,
        "top_results":        top_results,
        "processing_time_ms": elapsed_ms,
    }

    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2))
    background_tasks.add_task(_cleanup_old_results)

    return result


@app.get("/results/{result_id}/{filename}")
def serve_result_file(result_id: str, filename: str):
    # Prevent path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = OUTPUT_DIR / result_id / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(str(path), media_type="image/png")


@app.get("/results/{result_id}")
def get_result(result_id: str):
    json_path = OUTPUT_DIR / result_id / "prediction.json"
    if not json_path.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(json_path.read_text())
