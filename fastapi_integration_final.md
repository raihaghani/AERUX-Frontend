# AERUX FastAPI Integration Layer — Production-Ready Specification (v3 Final)

> **v3 is the definitive spec.** Merges all v2 production hardening with the full `preprocess_upload()` implementation (DICOM + NIfTI + .npy/.png) from the parallel review branch. Every previously identified issue is resolved.
>
> **All fixes included:** async blocking, image URL proxy, highlight_mask save, file validation (size + extension), TTL cleanup, LOCATION_LABELS refactor, prediction.json persistence, health versioning, `asyncio.get_running_loop()`, result_id lowercase enforcement, `preprocess_upload()` fully defined with DICOM/.nii/.nii.gz support, NIfTI suffix bug fixed.

---

## Architecture

```
Browser (Next.js)
   │  XHR / fetch  (never touches port 8000 directly)
   ▼
app/api/upload/route.ts              POST /api/upload  → forwards to FastAPI
app/api/images/[resultId]/[filename]/route.ts  GET     → proxies images from FastAPI
   │  HTTP (server-to-server only — no CORS issues)
   ▼
FastAPI  (port 8000, internal only)
   ├── POST /predict                  main inference endpoint
   ├── GET  /results/{id}             re-fetch stored prediction JSON
   ├── GET  /results/{id}/{filename}  serve generated image files
   └── GET  /health                   liveness + model info
```

---

## Step 0 — Create `Backend/constants.py` (do this first)

Move `LOCATION_LABELS` out of `train_multitask.py` (a heavy training script) into a shared
constants file. This prevents importing the entire training loop into the production server.

```python
# Backend/constants.py
LOCATION_LABELS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]
```

Then in both `train_multitask.py` and `infer_single_overlay.py`, replace the inline list with:

```python
from constants import LOCATION_LABELS
```

---

## Environment Variables

### `aerux-web/.env.local`
```bash
OPENAI_API_KEY=sk-...
FASTAPI_BASE_URL=http://localhost:8000   # server-side only, never exposed to browser
```

### `Backend/.env`
```bash
# NOTE: Always set FUSION_MODEL_PATH. load_fusion_model() has a hardcoded fallback glob
# rooted at E:/Education/Aerux_Final/outputs/ which will break on any other machine.
FUSION_MODEL_PATH=E:/Education/Aerux_Final/outputs/best_fusion_model.pth
UPLOAD_DIR=./uploads
OUTPUT_DIR=./inference_outputs
MAX_UPLOAD_MB=200
RESULT_TTL_HOURS=24
```

---

## `Backend/main.py` — Complete FastAPI Application

### Imports and config

```python
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

# ── Config ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR",  "./uploads"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR",  "./inference_outputs"))
FUSION_PATH = os.getenv("FUSION_MODEL_PATH")
MAX_BYTES   = int(os.getenv("MAX_UPLOAD_MB", 200)) * 1024 * 1024
TTL_HOURS   = int(os.getenv("RESULT_TTL_HOURS", 24))

# .nii/.nii.gz are now supported via preprocess_upload()
ALLOWED_EXT = {".dcm", ".npy", ".png", ".jpg", ".jpeg", ".nii"}
ALLOWED_MOD = {"CTA", "MRA", "MRI"}

device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model    = None
MODEL_LOADED_AT: str | None = None
```

### App startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global fusion_model, MODEL_LOADED_AT
    print(f"[startup] Loading fusion model: {FUSION_PATH}")
    fusion_model = load_fusion_model(FUSION_PATH, device)
    fusion_model.eval()
    MODEL_LOADED_AT = datetime.utcnow().isoformat() + "Z"
    print("[startup] Model ready.")
    yield
    # No explicit cleanup needed on shutdown

app = FastAPI(title="AERUX ML API", version="1.0.0", lifespan=lifespan)

# CORS: allow only the Next.js server — browsers never call FastAPI directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

### Helper 1 — `preprocess_upload()`

Handles all supported upload formats and converts to the `(vol_7ch, display_gray)` tuple
the inference pipeline expects. This is the only preprocessing entry point — all format
branching lives here.

```python
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
        vol_3d  = load_dicom_series(upload_path.parent)          # → 3D numpy (D, H, W)
        vol_25d = extract_2_5d_slices(vol_3d, num_slices=7)      # → (H, W, 7)
        vol_25d = resize_volume(vol_25d, (256, 256))              # → (256, 256, 7)
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
        vol_25d = normalize_volume(vol_25d)
        vol_7ch = np.transpose(vol_25d, (2, 0, 1)).astype(np.float32)
        display = (vol_7ch[3] * 255).astype(np.uint8)
        return vol_7ch, display

    else:
        raise HTTPException(
            400,
            f"Unsupported format '{suffix}'. Allowed: .dcm .nii .nii.gz .npy .png .jpg"
        )
```

---

### Helper 2 — `_run_inference_sync()`

All blocking PyTorch work lives here. Called via `run_in_executor` so it never blocks
the FastAPI event loop.

```python
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
```

---

### Helper 3 — `_cleanup_old_results()`

Runs as a FastAPI `BackgroundTask` after every `/predict` call. Deletes upload and output
directories older than `RESULT_TTL_HOURS`.

```python
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
```

---

## Endpoint 1 — `GET /health`

**Purpose:** Liveness + model readiness check. Called by the frontend on app load.

```python
@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_loaded":      fusion_model is not None,
        "model_loaded_at":   MODEL_LOADED_AT,
        "device":            str(device),
        "gpu_name":          (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
        "fusion_model_path": FUSION_PATH,
        "api_version":       "1.0.0",
    }
```

#### Response `200`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_loaded_at": "2026-03-25T13:00:00Z",
  "device": "cuda",
  "gpu_name": "NVIDIA Tesla T4",
  "fusion_model_path": "E:/Education/.../best_fusion_model.pth",
  "api_version": "1.0.0"
}
```

#### Response `503` (model still loading)
```json
{ "status": "error", "detail": "Model not loaded" }
```

---

## Endpoint 2 — `POST /predict`

### Request fields

| Field                  | Type     | Required | Default | Notes                                       |
|------------------------|----------|----------|---------|---------------------------------------------|
| `file`                 | File     | ✅       | —       | `.dcm` `.nii` `.nii.gz` `.npy` `.png` `.jpg` |
| `modality`             | string   | ✅       | —       | `"CTA"` `"MRA"` `"MRI"`                    |
| `sigma`                | float    | ❌       | `18.0`  | Gaussian blob spread for heatmap (pixels)  |
| `alpha`                | float    | ❌       | `0.45`  | Heatmap overlay transparency               |
| `threshold`            | float    | ❌       | `0.45`  | Binary mask cutoff on normalised heatmap   |
| `use_detection_weight` | bool     | ❌       | `true`  | Scale location probs by detection prob     |

### Handler

```python
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
    upload_dir  = UPLOAD_DIR / result_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / file.filename
    upload_path.write_bytes(file_bytes)

    # ── Preprocess ─────────────────────────────────────────────────────────
    # preprocess_upload() handles all formats: .npy/.png → load_input_as_7ch,
    # .dcm → DICOM pipeline, .nii/.nii.gz → SimpleITK pipeline
    try:
        vol_7ch, display_gray = preprocess_upload(upload_path, modality)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # ── Inference (non-blocking — runs in thread pool) ─────────────────────
    t0 = time.time()
    loop = asyncio.get_running_loop()   # get_running_loop() — correct for async context
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
```

### Response `200`
```json
{
  "result_id": "a3f2c1d9",
  "detection_prediction": 1,
  "detection_probabilities": {
    "no_aneurysm": 0.0712,
    "aneurysm": 0.9288
  },
  "top_3_locations": [
    { "label": "Anterior Communicating Artery",             "score": 0.871 },
    { "label": "Left Supraclinoid Internal Carotid Artery", "score": 0.643 },
    { "label": "Right Middle Cerebral Artery",              "score": 0.412 }
  ],
  "all_location_probabilities": {
    "Left Infraclinoid Internal Carotid Artery":  0.231,
    "Right Infraclinoid Internal Carotid Artery": 0.198,
    "Left Supraclinoid Internal Carotid Artery":  0.643,
    "Right Supraclinoid Internal Carotid Artery": 0.389,
    "Left Middle Cerebral Artery":                0.302,
    "Right Middle Cerebral Artery":               0.412,
    "Anterior Communicating Artery":              0.871,
    "Left Anterior Cerebral Artery":              0.187,
    "Right Anterior Cerebral Artery":             0.204,
    "Left Posterior Communicating Artery":        0.155,
    "Right Posterior Communicating Artery":       0.131,
    "Basilar Tip":                                0.093,
    "Other Posterior Circulation":                0.067
  },
  "image_urls": {
    "overlay":        "/results/a3f2c1d9/overlay.png",
    "heatmap":        "/results/a3f2c1d9/heatmap.png",
    "input_gray":     "/results/a3f2c1d9/input_gray.png",
    "highlight_mask": "/results/a3f2c1d9/highlight_mask.png"
  },
  "processing_time_ms": 1243
}
```

### Error responses

| Code  | When                                          |
|-------|-----------------------------------------------|
| `400` | Bad modality / unsupported extension / path traversal |
| `413` | File exceeds 200 MB limit                     |
| `422` | Missing required form field (FastAPI auto)    |
| `500` | Preprocessing or inference error              |
| `503` | Model not yet loaded (startup in progress)    |

---

## Endpoint 3 — `GET /results/{result_id}/{filename}`

**Purpose:** Serve generated image files. Called only by the Next.js image proxy — never directly by the browser.

```python
@app.get("/results/{result_id}/{filename}")
def serve_result_file(result_id: str, filename: str):
    # Prevent path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = OUTPUT_DIR / result_id / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(str(path), media_type="image/png")
```

---

## Endpoint 4 — `GET /results/{result_id}`

**Purpose:** Re-fetch a previously computed prediction JSON (e.g., user refreshes `/reports`).

```python
@app.get("/results/{result_id}")
def get_result(result_id: str):
    json_path = OUTPUT_DIR / result_id / "prediction.json"
    if not json_path.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(json_path.read_text())
```

---

## Frontend — New File: Next.js Image Proxy

**Create `aerux-web/app/api/images/[resultId]/[filename]/route.ts`**

Proxies image files from FastAPI to the browser, keeping port 8000 server-side only.

```typescript
// app/api/images/[resultId]/[filename]/route.ts
export async function GET(
  _req: Request,
  { params }: { params: { resultId: string; filename: string } }
) {
  const { resultId, filename } = params;

  // Sanitize — result_id is always 8 lowercase hex chars
  if (!resultId.match(/^[a-f0-9]{8}$/) || filename.includes("..")) {
    return new Response("Invalid request", { status: 400 });
  }

  const fastRes = await fetch(
    `${process.env.FASTAPI_BASE_URL}/results/${resultId}/${filename}`,
    { cache: "no-store" }
  );

  if (!fastRes.ok) {
    return new Response("Image not found", { status: fastRes.status });
  }

  const blob = await fastRes.blob();
  return new Response(blob, {
    headers: {
      "Content-Type":  "image/png",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
```

---

## Frontend — `app/api/upload/route.ts`

Replace the current stub with a real proxy that forwards to FastAPI and rewrites image URLs.

```typescript
// app/api/upload/route.ts
export async function POST(req: Request) {
  try {
    const form     = await req.formData();
    const file     = form.get("file")     as File;
    const modality = form.get("modality") as string;
    const sigma    = form.get("sigma")    as string | null;
    const alpha    = form.get("alpha")    as string | null;
    const threshold = form.get("threshold") as string | null;

    if (!file || !modality) {
      return new Response(
        JSON.stringify({ error: "file and modality are required" }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }

    // Forward to FastAPI
    const fastForm = new FormData();
    fastForm.append("file",     file);
    fastForm.append("modality", modality);
    if (sigma)     fastForm.append("sigma",     sigma);
    if (alpha)     fastForm.append("alpha",     alpha);
    if (threshold) fastForm.append("threshold", threshold);

    const fastRes = await fetch(
      `${process.env.FASTAPI_BASE_URL}/predict`,
      { method: "POST", body: fastForm }
    );

    const data = await fastRes.json();

    if (!fastRes.ok) {
      return new Response(
        JSON.stringify({ error: data.detail ?? "Inference failed" }),
        { status: fastRes.status, headers: { "content-type": "application/json" } }
      );
    }

    // Rewrite image_urls so the browser hits the Next.js proxy, not port 8000
    // /results/{id}/overlay.png  →  /api/images/{id}/overlay.png
    if (data.image_urls) {
      for (const key of Object.keys(data.image_urls)) {
        data.image_urls[key] = (data.image_urls[key] as string)
          .replace("/results/", "/api/images/");
      }
    }

    return new Response(
      JSON.stringify({ ok: true, ...data }),
      { status: 200, headers: { "content-type": "application/json" } }
    );

  } catch {
    return new Response(
      JSON.stringify({ error: "Upload failed" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }
}
```

---

## Frontend — Zustand Store (`store/ui.ts`)

Add prediction state to the existing store:

```typescript
type PredictionResult = {
  result_id:                  string;
  detection_prediction:       0 | 1;
  detection_probabilities:    { no_aneurysm: number; aneurysm: number };
  top_3_locations:            Array<{ label: string; score: number }>;
  all_location_probabilities: Record<string, number>;
  image_urls: {
    overlay:        string;
    heatmap:        string;
    input_gray:     string;
    highlight_mask: string;
  };
  processing_time_ms: number;
};

// Add to UIState interface:
selectedModality:      "CTA" | "MRA" | "MRI" | null;
predictionResult:      PredictionResult | null;
predictionError:       string | null;

// Add actions:
setSelectedModality:   (m: "CTA" | "MRA" | "MRI") => void;
setPredictionResult:   (r: PredictionResult | null) => void;
setPredictionError:    (e: string | null) => void;
```

---

## Frontend — `app/upload/page.tsx`

```typescript
const setSelectedModality = useUIStore((s) => s.setSelectedModality);
const setPredictionResult  = useUIStore((s) => s.setPredictionResult);
const setPredictionError   = useUIStore((s) => s.setPredictionError);
const setLoading           = useUIStore((s) => s.setLoading);

const [modality, setModality] = useState<"CTA" | "MRA" | "MRI">("CTA");

// Modality dropdown — place above the drop zone
<select
  value={modality}
  onChange={(e) => setModality(e.target.value as "CTA" | "MRA" | "MRI")}
  className="rounded-xl border px-3 py-2 text-sm text-[color:var(--aerux-navy)]"
>
  <option value="CTA">CTA — CT Angiography</option>
  <option value="MRA">MRA — MR Angiography</option>
  <option value="MRI">MRI — Magnetic Resonance</option>
</select>

function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file",     file);
  formData.append("modality", modality);

  setSelectedModality(modality);
  setPredictionError(null);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable)
      setUploadProgress(Math.round((e.loaded / e.total) * 100));
  };

  xhr.onloadstart = () => setLoading(true);

  xhr.onload = () => {
    setUploadProgress(100);
    const res = JSON.parse(xhr.responseText || "{}");
    if (res.ok) {
      setPredictionResult(res);
      setLoading(false);
      router.push("/visuals");
    } else {
      setPredictionError(res.error ?? "Inference failed");
      setLoading(false);
    }
    setTimeout(() => setUploadProgress(0), 800);
  };

  xhr.onerror = () => {
    setPredictionError("Network error");
    setLoading(false);
  };

  xhr.send(formData);
}
```

---

## Frontend — `app/visuals/page.tsx`

```typescript
const result = useUIStore((s) => s.predictionResult);

// Stats cards
const stats = result ? [
  { label: "Detection",    value: result.detection_prediction === 1 ? "Aneurysm Detected" : "Clear" },
  { label: "Confidence",   value: `${(result.detection_probabilities.aneurysm * 100).toFixed(1)}%` },
  { label: "Top Location", value: result.top_3_locations[0]?.label ?? "N/A" },
] : [];

// Overlay image — src goes through Next.js proxy, never port 8000
<img
  src={result.image_urls.overlay}
  alt="Heatmap overlay"
  className="w-full rounded-xl object-cover"
/>

// All 13 location probability bars, ranked highest first
Object.entries(result.all_location_probabilities)
  .sort(([, a], [, b]) => b - a)
  .map(([label, prob]) => (
    <div key={label} className="flex items-center gap-3">
      <span className="w-56 truncate text-xs text-zinc-600">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-zinc-100">
        <div
          style={{ width: `${prob * 100}%` }}
          className="h-1.5 rounded-full bg-[color:var(--aerux-accent)]"
        />
      </div>
      <span className="text-xs w-10 text-right">{(prob * 100).toFixed(0)}%</span>
    </div>
  ))
```

---

## Frontend — `app/reports/page.tsx`

```typescript
const result   = useUIStore((s) => s.predictionResult);
const modality = useUIStore((s) => s.selectedModality);

// Findings from real predictions
const findings = result
  ? result.top_3_locations.map((loc) => ({
      title:      loc.label,
      confidence: Math.round(loc.score * 100),
      note:       `Location probability: ${(loc.score * 100).toFixed(1)}%`,
    }))
  : [];

// In handleDownload() — embed real metadata into the PDF
page.drawText(`Modality: ${modality ?? "Unknown"}`, { x: 400, y: 690, size: 10, font });
page.drawText(
  `Aneurysm Confidence: ${((result?.detection_probabilities.aneurysm ?? 0) * 100).toFixed(1)}%`,
  { x: 50, y: 670, size: 10, font }
);
```

---

## File-to-Function Reference

| Backend file              | Function used in `main.py`                        | Purpose                          |
|---------------------------|---------------------------------------------------|----------------------------------|
| `constants.py`            | `LOCATION_LABELS`                                 | Map index → artery name          |
| `inference_fusion.py`     | `load_fusion_model(path, device)`                 | Model load at startup            |
| `infer_single_overlay.py` | `load_input_as_7ch(path, target_size)`            | .npy / .png → (7,256,256) tensor |
| `infer_single_overlay.py` | `build_location_prior_heatmap(...)`               | Generate location heatmap        |
| `infer_single_overlay.py` | `colorize_and_overlay(...)`                       | Generate visual images           |
| `preprocess_dataset.py`   | `load_dicom_series(series_dir)`                   | Load DICOM series from folder    |
| `preprocess_dataset.py`   | `extract_2_5d_slices(vol, num_slices=7)`          | 3D volume → 7-slice 2.5D         |
| `preprocess_dataset.py`   | `resize_volume(vol, (256, 256))`                  | Resize to model input size       |
| `preprocess_dataset.py`   | `normalize_volume(vol)`                           | Normalise to [0, 1]              |
| `SimpleITK` (pip)         | `sitk.ReadImage()` / `sitk.GetArrayFromImage()`   | NIfTI loading                    |

---

## Running the Stack

```bash
# Terminal 1 — FastAPI backend
cd d:/fyp/Backend
pip install fastapi uvicorn python-multipart SimpleITK
uvicorn main:app --reload --port 8000

# Terminal 2 — Next.js frontend
cd d:/fyp/AERUX-Frontend/aerux-web
npm run dev   # → http://localhost:3000
```

---

## Complete Request / Response Flow

```
User selects file + modality  →  /upload page
    │ XHR POST /api/upload  (Next.js)
    │   FormData: { file, modality, sigma, alpha, threshold }
    ▼
app/api/upload/route.ts  (Next.js server-side)
    │ Validates: file + modality present
    │ Forwards multipart → FastAPI POST /predict
    │ Rewrites image_urls: /results/{id}/*.png → /api/images/{id}/*.png
    ▼
FastAPI POST /predict
    ├─ Validates: modality ∈ {CTA,MRA,MRI}
    ├─ Validates: extension ∈ ALLOWED_EXT (incl. .nii.gz check)
    ├─ Validates: file size ≤ 200 MB
    ├─ Saves file → uploads/{id}/
    ├─ preprocess_upload(upload_path, modality)
    │       .npy / .png  → load_input_as_7ch()
    │       .dcm         → load_dicom_series() → extract_2_5d_slices() → resize → normalize
    │       .nii/.nii.gz → SimpleITK → extract_2_5d_slices() → resize → normalize
    │       → (vol_7ch: float32 (7,256,256), display_gray: uint8 (256,256))
    ├─ asyncio.get_running_loop().run_in_executor(_run_inference_sync)
    │       ├─ fusion_model(inputs, modality_indices)
    │       ├─ softmax(det_logits) → det_probs
    │       ├─ sigmoid(loc_logits) → loc_probs
    │       ├─ build_location_prior_heatmap(sigma)
    │       └─ colorize_and_overlay(alpha, threshold)
    ├─ Saves: overlay.png, heatmap.png, input_gray.png, highlight_mask.png
    ├─ Saves: prediction.json
    ├─ BackgroundTask: _cleanup_old_results()
    └─ Returns JSON
    ▼
Next.js rewrites image_urls → returns { ok: true, ...result } to browser
    ▼
Zustand: setPredictionResult(res)
router.push("/visuals")
    ▼
/visuals page
    • <img src="/api/images/{id}/overlay.png"> → Next.js proxy → FastAPI → disk
    • location bars: all_location_probabilities (13 entries, sorted)
    • stats: detection_prediction + aneurysm confidence %
    ▼
/reports page
    • findings cards: top_3_locations
    • PDF via pdf-lib: real modality + confidence + location data
```
