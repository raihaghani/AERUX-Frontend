import asyncio, json, os, shutil, time, uuid
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
from inference_fusion import load_fusion_model
from infer_single_overlay import (
    build_location_prior_heatmap,
    colorize_and_overlay,
    load_input_as_7ch,
)
from preprocess_dataset import (
    load_dicom_series,
    extract_2_5d_slices,
    resize_volume,
    normalize_volume,
)
from data.preprocessing import create_preprocessor

from dotenv import load_dotenv

# Load local .env first
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR",  "./uploads"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR",  "./inference_outputs"))
FUSION_PATH = os.getenv("FUSION_MODEL_PATH")
MAX_BYTES   = int(os.getenv("MAX_UPLOAD_MB", 200)) * 1024 * 1024
TTL_HOURS   = int(os.getenv("RESULT_TTL_HOURS", 24))

# Ensure required dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# .nii/.nii.gz are now supported via preprocess_upload()
ALLOWED_EXT = {".dcm", ".npy", ".png", ".jpg", ".jpeg", ".nii"}
ALLOWED_MOD = {"CTA", "MRA", "MRI"}

device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model    = None
MODEL_LOADED_AT: str | None = None


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

# CORS: allow only the Next.js server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def preprocess_upload(upload_path: Path, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert any supported upload format → (vol_7ch, display_gray).

    Returns:
        vol_7ch      — float32 numpy array, shape (7, 256, 256), values in [0, 1]
        display_gray — uint8 numpy array,  shape (256, 256),     values in [0, 255]

    Supported formats:
        .npy / .png / .jpg / .jpeg  → load_input_as_7ch() from infer_single_overlay.py
        .dcm                        → load_dicom_series() + extract_2_5d_slices() pipeline
        .nii / .nii.gz              → SimpleITK → same 2.5D pipeline as DICOM
    """
    filename_lower = upload_path.name.lower()
    suffix = upload_path.suffix.lower()

    # ── .npy / .png / .jpg ────────────────────────────────────────────────────
    if suffix in (".npy", ".png", ".jpg", ".jpeg"):
        return load_input_as_7ch(str(upload_path), target_size=256)

    # ── DICOM (.dcm) ──────────────────────────────────────────────────────────
    elif suffix == ".dcm":
        # load_dicom_series expects the parent directory containing the full series
        vol_3d, _meta, _sop_map = load_dicom_series(upload_path.parent)  # → 3D numpy (N, H, W)
        vol_25d = extract_2_5d_slices(vol_3d, num_slices=7)      # → (H, W, 7)
        vol_25d = resize_volume(vol_25d, (256, 256))              # → (256, 256, 7)
        
        prep = create_preprocessor(modality)
        if prep is not None:
            vol_25d = np.stack(
                [prep.preprocess(vol_25d[:, :, i])["preprocessed"] for i in range(7)],
                axis=-1,
            )

        vol_25d = normalize_volume(vol_25d)                       # → [0, 1]
        vol_7ch = np.transpose(vol_25d, (2, 0, 1)).astype(np.float32)  # → (7, 256, 256)
        display = (vol_7ch[3] * 255).astype(np.uint8)            # centre slice for overlay
        return vol_7ch, display

    # ── NIfTI (.nii or .nii.gz) ───────────────────────────────────────────────
    # Note: suffix alone is ".gz" for .nii.gz, so we check the full filename.
    elif suffix == ".nii" or filename_lower.endswith(".nii.gz"):
        import SimpleITK as sitk
        img    = sitk.ReadImage(str(upload_path))
        vol_3d = sitk.GetArrayFromImage(img).astype(np.float32)  # → (D, H, W)
        vol_25d = extract_2_5d_slices(vol_3d, num_slices=7)
        vol_25d = resize_volume(vol_25d, (256, 256))
        
        prep = create_preprocessor(modality)
        if prep is not None:
             vol_25d = np.stack(
                [prep.preprocess(vol_25d[:, :, i])["preprocessed"] for i in range(7)],
                axis=-1,
            )

        vol_25d = normalize_volume(vol_25d)
        vol_7ch = np.transpose(vol_25d, (2, 0, 1)).astype(np.float32)
        display = (vol_7ch[3] * 255).astype(np.uint8)
        return vol_7ch, display

    else:
        raise HTTPException(
            400,
            f"Unsupported format '{suffix}'. Allowed: .dcm .nii .nii.gz .npy .png .jpg"
        )


def _run_inference_sync(
    vol_7ch: np.ndarray,
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

    det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]   # [p_neg, p_pos]
    loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]       # [p_0 .. p_12]
    det_pred  = int(np.argmax(det_probs))
    det_prob  = float(det_probs[1])

    heat, loc_weights = build_location_prior_heatmap(
        det_prob, loc_probs, 256, 256, sigma, use_detection_weight
    )
    heat_uint8, binary_mask, heat_color, overlay = colorize_and_overlay(
        None, heat, alpha, threshold
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
    modality:             str        = Form(...),
    sigma:                float      = Form(18.0),
    alpha:                float      = Form(0.45),
    threshold:            float      = Form(0.45),
    use_detection_weight: bool       = Form(True),
):
    if fusion_model is None:
        raise HTTPException(503, "Model not yet loaded or missing FUSION_MODEL_PATH configuration")

    # ── Validate modality ──────────────────────────────────────────────────
    if modality not in ALLOWED_MOD:
        raise HTTPException(400, f"modality must be one of {ALLOWED_MOD}")

    # ── Validate file extension ────────────────────────────────────────────
    filename_lower = (file.filename or "").lower()
    ext = Path(filename_lower).suffix

    # .nii.gz needs special handling — suffix alone is ".gz"
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
    # Force lowercase hex for result_id — keeps image proxy regex consistent
    result_id   = str(uuid.uuid4())[:8].lower()
    
    # Store the file using its real filename to preserve extensions correctly for preprocessing
    upload_dir  = UPLOAD_DIR / result_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    real_filename = file.filename if file.filename else f"upload{ext}"
    upload_path = upload_dir / real_filename
    upload_path.write_bytes(file_bytes)

    # ── Preprocess ─────────────────────────────────────────────────────────
    # preprocess_upload() handles all formats
    try:
        vol_7ch, display_gray = preprocess_upload(upload_path, modality)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

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
            vol_7ch, modality, sigma, alpha, threshold, use_detection_weight,
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

    # Persist JSON — enables GET /results/{id} to work on re-visits
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2))

    # Schedule TTL cleanup (non-blocking)
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
