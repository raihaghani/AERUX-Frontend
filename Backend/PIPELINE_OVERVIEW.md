# Intracranial Aneurysm Detection, Localization & Segmentation Pipeline

## Complete Technical Overview

**Project:** Aerux — AI-Powered Intracranial Aneurysm Detection System  
**Dataset:** RSNA Intracranial Aneurysm Detection 
**Modalities:** CTA, MRA, MRI
**Tasks:** Binary Detection, 13-Location Localization

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Phase 1 — Data Preprocessing (DICOM to 2.5D)](#2-phase-1--data-preprocessing-dicom-to-25d)
3. [Phase 2 — Multi-Task Model Training (Per-Modality)](#3-phase-2--multi-task-model-training-per-modality)
4. [Phase 3 — Multi-Modal Fusion Training](#4-phase-3--multi-modal-fusion-training)
5. [Phase 4 — 3D Segmentation (SwinUNETR)](#5-phase-4--3d-segmentation-swinunetr)
6. [Phase 5 — Inference Pipeline](#6-phase-5--inference-pipeline)
7. [Output Formats](#7-output-formats)
8. [Web Application Integration](#8-web-application-integration)
9. [API Design for Web Deployment](#9-api-design-for-web-deployment)
10. [File & Directory Reference](#10-file--directory-reference)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AERUX PIPELINE OVERVIEW                         │
└─────────────────────────────────────────────────────────────────────────┘

  DICOM Series                                              Web Interface
  (CTA/MRA/MRI)                                            ┌────────────┐
       │                                                    │  Upload    │
       ▼                                                    │  Image     │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │            │
│  PREPROCESS  │───▶│  2.5D .npy   │───▶│   MULTI-TASK │   │  Results:  │
│  - HU Window │    │  (256×256×7) │    │   ResNet50   │   │  - Yes/No  │
│  - N4 Bias   │    └──────────────┘    │  (per modal) │   │  - Location│
│  - Vesselness│                        └──────┬───────┘   │  - Overlay │
│  - 2.5D Slice│                               │           └─────▲──────┘
└──────────────┘                               ▼                 │
                                        ┌──────────────┐         │
                                        │   ATTENTION   │─────────┘
                                        │   FUSION      │
                                        │   MODEL       │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │   INFERENCE   │
                                        │  - Detection  │
                                        │  - Location   │
                                        │  - Heatmap    │
                                        │  - Overlay    │
                                        └──────────────┘
```

---

## 2. Phase 1 — Data Preprocessing (DICOM to 2.5D)

### Overview

Raw medical images arrive as DICOM series (a folder of `.dcm` files per patient scan). The preprocessing pipeline converts each 3D volume into a compact **2.5D representation** — 7 representative axial slices stacked as channels — suitable for 2D CNN processing.

### Script: `preprocess_dataset.py`

### Input

| Item | Detail |
|------|--------|
| **Format** | DICOM files (`.dcm`) organized as `series/{SeriesInstanceUID}/*.dcm` |
| **Metadata** | `train.csv` — labels (Aneurysm Present, 13 location columns, Modality) |
| **Localizers** | `train_localizers.csv` — SOPInstanceUIDs marking aneurysm-containing slices |
| **Error List** | `error_data.yaml` — series IDs to exclude (corrupted/unusable data) |

### Step-by-Step Process

#### Step 1: DICOM Loading & Sorting
```
Raw DICOM folder ──▶ Read all .dcm files
                     Apply RescaleSlope/Intercept (→ approximate HU for CT)
                     Sort by ImagePositionPatient[2] (Z-axis)
                     Result: ordered 3D volume
```

#### Step 2: 2.5D Slice Extraction
The 2.5D approach reduces a full 3D volume to exactly **7 axial slices**, preserving spatial context while enabling use of efficient 2D CNNs.

```
Full 3D Volume (e.g., 200+ slices)
       │
       ├── If aneurysm labels exist:
       │     Find center_slice from train_localizers.csv (median of labeled SOPs)
       │     Extract slices: [center-3, center-2, center-1, center, center+1, center+2, center+3]
       │     (clamped to volume bounds, zero-padded if needed)
       │
       └── If no labels / normal case:
             Pick 7 evenly-spaced slices across the full depth
```

#### Step 3: Modality-Specific Preprocessing

**CTA (CT Angiography):**
```
Raw HU values ──▶ HU Windowing (center=40, width=80) ──▶ [Sato Vesselness Filter]
                  Clips to 0-80 HU range                   Enhances tubular
                  Normalizes to [0, 1]                      blood vessel structures
                  Focuses on brain tissue + vessels          Scales: 1-8, step 2
```

**MRI / MRA:**
```
Raw intensities ──▶ N4 Bias Field Correction ──▶ [Sato Vesselness Filter]
                    Corrects B-field                Enhances blood vessels
                    intensity non-uniformity         Same multi-scale approach
                    via SimpleITK (4-level,          as CTA
                    50 iterations each)
```

#### Step 4: Resize & Save
```
7 processed slices ──▶ Resize each to 256×256 ──▶ Stack as (H, W, 7) ──▶ Save as float32 .npy
```

### Output

| Item | Detail |
|------|--------|
| **Format** | NumPy `.npy` files, dtype `float32` |
| **Shape** | `(256, 256, 7)` — height × width × slices |
| **Path** | `Preprocessed_images_2.5D/{CTA\|MRA\|MRI}/{aneurysm\|no_aneurysm}/{SeriesUID}.npy` |

### Preprocessing Configuration (`config/config.py`)

| Parameter | CTA | MRI/MRA |
|-----------|-----|---------|
| HU Windowing | center=40, width=80 | N/A |
| N4 Bias Correction | No | Yes (4-level, 50 iter) |
| CLAHE | Optional (disabled) | N/A |
| Sato Vesselness | Yes, scales 1-8, step 2 | Yes, scales 1-8, step 2 |
| Otsu Thresholding | N/A | Optional (disabled) |
| Target Size | 256×256 | 256×256 |
| Num Slices | 7 | 7 |

---

## 3. Phase 2 — Multi-Task Model Training (Per-Modality)

### Overview

A single **MultiTaskResNet50** architecture simultaneously performs three tasks from one shared backbone. Three independent models are trained — one for CTA, one for MRA, one for MRI — each on its own modality-specific subset.

### Script: `train_multitask.py`

### Architecture: MultiTaskResNet50

```
Input: (B, 7, 256, 256)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  MODIFIED ResNet50 ENCODER                                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Conv1: 7→64, 7×7, stride 2    ──▶ 128×128            │   │
│  │ BN + ReLU + MaxPool            ──▶ 64×64              │   │
│  │ Layer1: 64→256                 ──▶ 64×64    (skip 1)  │   │
│  │ Layer2: 256→512, stride 2      ──▶ 32×32    (skip 2)  │   │
│  │ Layer3: 512→1024, stride 2     ──▶ 16×16    (skip 3)  │   │
│  │ Layer4: 1024→2048, stride 2    ──▶ 8×8                │   │
│  └───────────────────────────────────────────────────────┘   │
│                         │                                     │
│              ┌──────────┴──────────┐                          │
│              ▼                     ▼                          │
│  ┌──────────────────┐   ┌──────────────────────────────────┐ │
│  │  Global Avg Pool  │   │  U-NET STYLE DECODER             │ │
│  │  Flatten → 2048-D │   │                                  │ │
│  │         │         │   │  8×8 ─▶UpConv+Skip─▶ 16×16      │ │
│  │    ┌────┴────┐    │   │  16×16─▶UpConv+Skip─▶ 32×32     │ │
│  │    ▼         ▼    │   │  32×32─▶UpConv+Skip─▶ 64×64     │ │
│  │ ┌──────┐ ┌──────┐│   │  64×64─▶UpConv+Skip─▶ 128×128   │ │
│  │ │DETECT│ │LOCATE ││   │  128×128─▶UpConv ──▶ 256×256    │ │
│  │ │Head  │ │Head   ││   │                                  │ │
│  │ │2048  │ │2048   ││   │  Output: (B, 1, 256, 256)       │ │
│  │ │→512  │ │→512   ││   │  (segmentation mask logits (not used))│ │
│  │ │→2    │ │→13    ││   └──────────────────────────────────┘ │
│  │ └──────┘ └──────┘│                                        │
│  └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘

Output:  detection_logits (B, 2)       ─ Binary: aneurysm present?
         localization_logits (B, 13)   ─ Multi-label: which of 13 locations?
         segmentation_mask (B, 1, 256, 256) ─ Pixel-level mask logits (wrong predictions)
```

### Three Task Heads

| Head | Input Dim | Output | Activation (at inference) | Loss |
|------|-----------|--------|--------------------------|------|
| **Detection** | 2048 → 512 → 2 | Binary (aneurysm yes/no) | Softmax | CrossEntropyLoss |
| **Localization** | 2048 → 512 → 13 | Multi-label (13 locations) | Sigmoid (per location) | BCEWithLogitsLoss |
| **Segmentation** | Decoder skip-connections → (1, 256, 256) | Pixel mask | Sigmoid | Dice + BCE combined |

### 13 Anatomical Locations

| # | Location |
|---|----------|
| 1 | Left Infraclinoid Internal Carotid Artery |
| 2 | Right Infraclinoid Internal Carotid Artery |
| 3 | Left Supraclinoid Internal Carotid Artery |
| 4 | Right Supraclinoid Internal Carotid Artery |
| 5 | Left Middle Cerebral Artery |
| 6 | Right Middle Cerebral Artery |
| 7 | Anterior Communicating Artery |
| 8 | Left Anterior Cerebral Artery |
| 9 | Right Anterior Cerebral Artery |
| 10 | Left Posterior Communicating Artery |
| 11 | Right Posterior Communicating Artery |
| 12 | Basilar Tip |
| 13 | Other Posterior Circulation |

### Multi-Task Loss

```
L_total = 1.0 × L_detection + 1.0 × L_localization + 2.0 × L_segmentation
```

- Segmentation is weighted 2× because it requires more gradient signal to learn spatial masks.
- The combined loss trains all three heads jointly through a shared encoder.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet50 (ImageNet pretrained) |
| First Conv | Modified: 7-channel input (pretrained weights tiled from 3-ch) |
| Cross-Validation | 4-fold stratified |
| Optimizer | Adam (lr=1e-4, weight_decay=5e-5) |
| Scheduler | CosineAnnealingLR |
| Epochs | 100 (early stopping, patience 15) |
| Batch Size | 8 |
| Mixed Precision | AMP enabled |
| Metric | Mean Weighted Columnwise AUC-ROC |
| Augmentation | Random flips (H/V), 90° rotations |

### Evaluation Metric (Competition Metric)

```
Final Score = 1/2 × (AUC_detection + 1/13 × Σ AUC_location_i)
```

### Saved Artifacts (per modality, per fold)

- `best_model_{MODALITY}_fold{k}.pth` — model state dict + optimizer + metrics
- `history_fold{k}.json` — training/validation loss and metrics per epoch
- `fold{k}_results.json` — final fold evaluation
- `cross_validation_results_{MODALITY}.json` — aggregated CV results
- PNG plots: training curves, confusion matrix, per-location AUC bar charts

---

## 4. Phase 3 — Multi-Modal Fusion Training

### Overview

After training three separate modality models, the **SimpleFusionModel** learns to combine their features using an **attention mechanism**. The three base models are **frozen** (no weight updates) — only the attention and fusion heads are trained.

### Script: `train_fusion.py`

### Architecture: SimpleFusionModel (Attention-Based Fusion)

```
┌───────────────────────────────────────────────────────────────────┐
│                   FUSION MODEL ARCHITECTURE                       │
│                                                                   │
│  Input Sample (only ONE modality per sample)                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │  CTA Input ──▶ [Frozen CTA ResNet50 Encoder] ──▶ 2048-D ─┐  │ │
│  │                                                            │  │ │
│  │  MRA Input ──▶ [Frozen MRA ResNet50 Encoder] ──▶ 2048-D ─┤  │ │
│  │                                (only one active)           │  │ │
│  │  MRI Input ──▶ [Frozen MRI ResNet50 Encoder] ──▶ 2048-D ─┘  │ │
│  │                                                   │          │ │
│  └───────────────────────────────────────────────────┼──────────┘ │
│                                                      ▼            │
│                              ┌──────────────────────────────┐     │
│                              │   LEARNED ATTENTION MECHANISM │     │
│                              │                              │     │
│                              │   Q = Linear(2048 → 256)     │     │
│                              │   K = Linear(2048 → 256)     │     │
│                              │   V = Linear(2048 → 256)     │     │
│                              │                              │     │
│                              │   Attention = softmax(QKᵀ/√d)│     │
│                              │   Output = Attention × V     │     │
│                              │   Result: 256-D features     │     │
│                              └──────────────┬───────────────┘     │
│                                             │                     │
│                     ┌───────────────────────┼───────────────┐     │
│                     ▼                       ▼               ▼     │
│              ┌────────────┐         ┌────────────┐   ┌──────────┐ │
│              │ DETECTION  │         │ LOCATION   │   │ SEGMENT  │ │
│              │ 256→128→2  │         │ 256→128→   │   │ 256→128  │ │
│              │            │         │ 64→13      │   │ →64→     │ │
│              │ (2 logits) │         │ (13 logits)│   │ Upsample │ │
│              └────────────┘         └────────────┘   │ 32→256²  │ │
│                                                      └──────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Why Attention-Based Fusion?

1. **Leverages pre-trained models** — no need to retrain from scratch
2. **Adaptive feature weighting** — learns modality-specific importance
3. **Unified predictions** — all 2,000 samples go through one fusion model
4. **Interpretability** — attention weights show which features matter most

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Models | 3 frozen MultiTaskResNet50 (CTA, MRA, MRI) |
| Trainable Parameters | Attention Q/K/V + fusion heads only |
| Data Split | 70% train / 15% val / 15% test (stratified) |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Loss Weights | Detection: 1.0, Location: 2.5, Segmentation: 0.5 |

### Saved Artifacts

- `best_fusion_model.pth` — full fusion state dict + base model paths + metadata
- `test_results_fusion.json` — final test metrics
- Training/validation/test visualization plots

---

## 5. Phase 4 — Inference Pipeline

### 6A. Batch Inference (`inference_fusion.py`)

Runs the fusion model on an entire dataset split (typically the test set).

```
CSV (test split)
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│ Load .npy files  │────▶│ Load Fusion Model│
│ per modality     │     │ + 3 base models  │
└──────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Forward Pass     │
                         │ - Softmax det    │
                         │ - Sigmoid loc    │
                         │ - Sigmoid seg    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
            predictions.npz  predictions.csv  inference_summary.json
```

### 6B. Single-Image Inference with Overlay (`infer_single_overlay.py`)

This is the **primary inference entry point** for individual images and the basis for web integration.

```
Single .npy or image
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Input                                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ .npy file (preferred):                                     │  │
│  │   Load → ensure shape (7, 256, 256) → extract center       │  │
│  │   slice [3] for display                                    │  │
│  │                                                            │  │
│  │ .png/.jpg (convenience):                                   │  │
│  │   Load → grayscale → resize 256×256 → tile to 7 channels  │  │
│  │   (less reliable than true 2.5D input)                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  STEP 2: Model Forward Pass                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Tensor (1, 7, 256, 256) → Fusion Model                    │  │
│  │                                                            │  │
│  │ Outputs:                                                   │  │
│  │   detection_logits → softmax → P(aneurysm), P(no_aneurysm)│  │
│  │   location_logits  → sigmoid → 13 location probabilities   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  STEP 3: Build Weak Localization Heatmap                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ For each of 13 anatomical locations:                       │  │
│  │   - Look up fixed (x, y) anchor on 256×256 canvas         │  │
│  │   - Weight = location_prob × detection_prob                │  │
│  │   - Draw Gaussian blob (σ=18px) at anchor, scaled by weight│  │
│  │   - Sum all 13 blobs → raw heatmap                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  STEP 4: Generate Visual Outputs                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ - Normalize heatmap to [0, 255]                            │  │
│  │ - Apply JET colormap                                       │  │
│  │ - Blend with grayscale input (alpha=0.45)                  │  │
│  │ - Threshold normalized heatmap (≥0.45) → binary mask       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  STEP 5: Save Results                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ prediction.json    ─ detection + location probabilities    │  │
│  │ input_gray.png     ─ grayscale center slice                │  │
│  │ heatmap.png        ─ JET-colored heatmap                   │  │
│  │ highlight_mask.png ─ binary mask of high-probability areas │  │
│  │ overlay.png        ─ heatmap blended on input image        │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Anatomical Location Anchors (Normalized Coordinates)

Each location is mapped to a fixed anatomical position on the 256×256 canvas. These anchors represent typical anatomical positions of each artery on an axial brain scan:

| Location | Anchor (x, y) |
|----------|---------------|
| Left Infraclinoid ICA | (0.40, 0.62) |
| Right Infraclinoid ICA | (0.60, 0.62) |
| Left Supraclinoid ICA | (0.42, 0.53) |
| Right Supraclinoid ICA | (0.58, 0.53) |
| Left MCA | (0.30, 0.46) |
| Right MCA | (0.70, 0.46) |
| Anterior Communicating | (0.50, 0.40) |
| Left ACA | (0.43, 0.36) |
| Right ACA | (0.57, 0.36) |
| Left PComm | (0.43, 0.57) |
| Right PComm | (0.57, 0.57) |
| Basilar Tip | (0.50, 0.55) |
| Other Posterior | (0.50, 0.70) |

### Command-Line Usage

```bash
# Preferred: preprocessed .npy input
python infer_single_overlay.py \
    --fusion_model outputs/fusion_model/best_fusion_model.pth \
    --input Preprocessed_images_2.5D/CTA/aneurysm/SERIES_UID.npy \
    --modality CTA \
    --out_dir outputs/single_overlay

# Convenience: 2D image input (less accurate)
python infer_single_overlay.py \
    --fusion_model outputs/fusion_model/best_fusion_model.pth \
    --input patient_scan.png \
    --modality CTA \
    --out_dir outputs/single_overlay
```

---

## 7. Output Formats

### prediction.json (Single Image Inference)

```json
{
  "input_path": "path/to/input.npy",
  "modality": "CTA",
  "detection_prediction": 1,
  "detection_probabilities": {
    "no_aneurysm": 0.1234,
    "aneurysm": 0.8766
  },
  "top_3_locations": [
    { "label": "Left Middle Cerebral Artery", "score": 0.7234 },
    { "label": "Anterior Communicating Artery", "score": 0.5891 },
    { "label": "Right Supraclinoid Internal Carotid Artery", "score": 0.3456 }
  ],
  "all_location_probabilities": {
    "Left Infraclinoid Internal Carotid Artery": 0.12,
    "Right Infraclinoid Internal Carotid Artery": 0.08,
    "...": "..."
  },
  "overlay_note": "Weak localization from location probabilities (not true segmentation)."
}
```

### predictions.csv (Batch Inference)

| Column | Description |
|--------|-------------|
| SeriesInstanceUID | Unique series ID |
| Predicted_Aneurysm | 0 or 1 |
| Aneurysm_Probability | Float [0, 1] |
| {Location}_Prob (×13) | Per-location probability |

### predictions.npz (Batch Inference)

| Array | Shape | Description |
|-------|-------|-------------|
| detection_probs | (N, 2) | Softmax probabilities |
| detection_preds | (N,) | Argmax predictions |
| location_probs | (N, 13) | Sigmoid location probabilities |
| segmentation_preds | (N, 1, 256, 256) | Sigmoid segmentation masks |

### Visual Outputs (per sample)

| File | Content |
|------|---------|
| `input_gray.png` | Grayscale center slice (slice 4 of 7) |
| `heatmap.png` | JET-colored location-prior heatmap |
| `highlight_mask.png` | Binary mask thresholded from heatmap |
| `overlay.png` | Heatmap blended onto input (alpha=0.45) |

---

## 8. Web Application Integration

### How the Existing Pipeline Serves a Web Application

The trained models and inference scripts can be wrapped into a web application backend to provide three key clinical outputs:

### Output 1: Aneurysm Detection (Present / Absent)

```
User uploads DICOM/image ──▶ Preprocess to 2.5D .npy
                                    │
                                    ▼
                            Fusion Model Forward Pass
                                    │
                                    ▼
                            softmax(detection_logits)
                                    │
                                    ▼
                            P(aneurysm) > 0.5 → "ANEURYSM DETECTED"
                            P(aneurysm) ≤ 0.5 → "NO ANEURYSM FOUND"
```

**What the user sees:** A clear YES/NO result with a confidence percentage (e.g., "Aneurysm Detected — 87.6% confidence").

### Output 2: Aneurysm Location

```
Fusion Model ──▶ sigmoid(localization_logits) ──▶ 13 location probabilities
                                                         │
                                                         ▼
                                                 Rank by probability
                                                         │
                                                         ▼
                                                 "Top 3 Locations:
                                                  1. Left MCA (72.3%)
                                                  2. AComm (58.9%)
                                                  3. Right Supraclinoid ICA (34.6%)"
```

**What the user sees:** A ranked list of likely aneurysm locations with probabilities, optionally displayed on an anatomical brain diagram.

### Output 3: Mask Overlay Visualization

```
Location probabilities + Detection probability
                │
                ▼
        Build location-prior heatmap
        (Gaussian blobs at anatomical anchors,
         weighted by location × detection probs)
                │
                ▼
        Apply JET colormap ──▶ Blend with input image
                                        │
                                        ▼
                                 Overlay image showing
                                 suspected aneurysm regions
```

**What the user sees:** The original brain scan with a color-coded heatmap overlay highlighting regions where the aneurysm is most likely located. Red = high probability, Blue = low probability.

### End-to-End Web Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          WEB APPLICATION                                │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────────────────┐   │
│  │  FRONTEND    │    │  BACKEND (Python API Server)                 │   │
│  │  (React/     │    │                                              │   │
│  │   Next.js)   │    │  1. Receive uploaded file                    │   │
│  │              │    │  2. Detect modality (CTA/MRA/MRI)            │   │
│  │  ┌────────┐  │    │  3. Preprocess:                              │   │
│  │  │Upload  │──┼───▶│     - DICOM → read + sort slices            │   │
│  │  │DICOM / │  │    │     - Apply HU/N4/vesselness per modality   │   │
│  │  │Image   │  │    │     - Extract 7 slices → 256×256×7          │   │
│  │  └────────┘  │    │  4. Model inference:                         │   │
│  │              │    │     - Load fusion model (cached in memory)   │   │
│  │  ┌────────┐  │    │     - Forward pass → det, loc, seg logits   │   │
│  │  │Results │◀─┼────│  5. Post-process:                            │   │
│  │  │Display │  │    │     - Compute probabilities                  │   │
│  │  │        │  │    │     - Build heatmap from location priors     │   │
│  │  │• Yes/No│  │    │     - Generate overlay image                 │   │
│  │  │• Score │  │    │  6. Return JSON + images                     │   │
│  │  │• Locs  │  │    │                                              │   │
│  │  │• Image │  │    └──────────────────────────────────────────────┘   │
│  │  └────────┘  │                                                       │
│  └─────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. API Design for Web Deployment

### Recommended Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend API** | FastAPI (Python) or Flask |
| **Model Serving** | PyTorch (load once at startup, keep in GPU memory) |
| **Frontend** | React / Next.js / Streamlit (for quick prototype) |
| **File Handling** | `python-multipart` for uploads, `pydicom` for DICOM parsing |
| **Image Output** | Base64-encoded PNGs in JSON response, or serve via static URLs |

### REST API Endpoints

#### `POST /api/predict`

**Request:**
```
Content-Type: multipart/form-data

Fields:
  file: <uploaded DICOM folder or .npy file>
  modality: "CTA" | "MRA" | "MRI"
```

**Response:**
```json
{
  "success": true,
  "detection": {
    "prediction": "aneurysm",
    "confidence": 0.876,
    "probabilities": {
      "aneurysm": 0.876,
      "no_aneurysm": 0.124
    }
  },
  "locations": {
    "top_3": [
      { "name": "Left Middle Cerebral Artery", "probability": 0.723 },
      { "name": "Anterior Communicating Artery", "probability": 0.589 },
      { "name": "Right Supraclinoid ICA", "probability": 0.346 }
    ],
    "all": { ... }
  },
  "images": {
    "input_gray": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "highlight_mask": "data:image/png;base64,..."
  }
}
```

### Backend Pseudocode (FastAPI)

```python
from fastapi import FastAPI, UploadFile, Form
import torch
from infer_single_overlay import load_input_as_7ch, build_location_prior_heatmap, colorize_and_overlay
from inference_fusion import load_fusion_model

app = FastAPI()

# Load model ONCE at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fusion_model("path/to/best_fusion_model.pth", device)

@app.post("/api/predict")
async def predict(file: UploadFile, modality: str = Form(...)):
    # 1. Save uploaded file temporarily
    temp_path = save_temp_file(file)

    # 2. If DICOM, preprocess to 2.5D .npy
    if is_dicom(temp_path):
        npy_data = preprocess_dicom_to_2_5d(temp_path, modality)
    else:
        npy_data = load_input_as_7ch(temp_path)

    # 3. Run model inference
    vol_7ch, display_gray = npy_data
    x = torch.from_numpy(vol_7ch).unsqueeze(0).to(device)

    inputs = {"CTA": None, "MRA": None, "MRI": None}
    modality_indices = {"CTA": [], "MRA": [], "MRI": []}
    inputs[modality] = x
    modality_indices[modality] = [0]

    with torch.no_grad():
        det_logits, loc_logits, _ = model(inputs, modality_indices)
        det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]
        loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]

    # 4. Build heatmap and overlay
    heat, weights = build_location_prior_heatmap(
        det_prob=float(det_probs[1]),
        location_probs=loc_probs,
        h=256, w=256, sigma=18.0
    )
    heat_uint8, binary_mask, heat_color, overlay = colorize_and_overlay(
        display_gray, heat, alpha=0.45, threshold=0.45
    )

    # 5. Encode images as base64 and return JSON
    return build_response(det_probs, loc_probs, display_gray, heat_color, overlay, binary_mask)
```

### Preprocessing Helper for DICOM Upload

When a user uploads raw DICOM files through the web interface, the backend must replicate the offline preprocessing:

```python
def preprocess_dicom_to_2_5d(dicom_folder, modality):
    """Convert uploaded DICOM to 2.5D representation for model input."""

    # 1. Read and sort DICOM slices by Z position
    slices = read_and_sort_dicoms(dicom_folder)

    # 2. Apply rescale slope/intercept to get proper HU values
    volume_3d = apply_rescale(slices)

    # 3. Extract 7 evenly-spaced slices (no localizer info for new patients)
    slices_7 = extract_evenly_spaced(volume_3d, num_slices=7)

    # 4. Resize each slice to 256×256
    slices_7_resized = [cv2.resize(s, (256, 256)) for s in slices_7]

    # 5. Apply modality-specific preprocessing
    if modality == "CTA":
        preprocessor = CTAPreprocessor(apply_hu_windowing=True, apply_vesselness=True)
        slices_7_processed = [preprocessor.preprocess(s)['preprocessed'] for s in slices_7_resized]
    elif modality in ("MRA", "MRI"):
        preprocessor = MRIPreprocessor(apply_n4=True, apply_vesselness=True)
        slices_7_processed = [preprocessor.preprocess(s)['preprocessed'] for s in slices_7_resized]

    # 6. Stack into (7, 256, 256) tensor
    volume_2_5d = np.stack(slices_7_processed, axis=0).astype(np.float32)

    return volume_2_5d, generate_display_image(volume_2_5d)
```

### Model Files Required for Deployment

| File | Size (approx.) | Purpose |
|------|----------------|---------|
| `best_fusion_model.pth` | ~300 MB | Fusion model + references to base models |
| `best_model_CTA_fold4.pth` | ~100 MB | CTA base encoder |
| `best_model_MRA_fold4.pth` | ~100 MB | MRA base encoder |
| `best_model_MRI_fold4.pth` | ~100 MB | MRI base encoder |

**Total model storage: ~600 MB**

### Hardware Requirements

| Scenario | Recommendation |
|----------|---------------|
| **Development/Testing** | CPU-only (slower, ~2-5s per image) |
| **Production** | GPU with ≥4 GB VRAM (NVIDIA T4/RTX 2060+), ~0.1-0.5s per image |
| **Cloud Deployment** | AWS EC2 `g4dn.xlarge` or GCP `n1-standard-4` with T4 GPU |

---

## 10. File & Directory Reference

### Project Structure

```
Aerux_Final/
├── config/
│   └── config.py                    # Central configuration (all hyperparameters)
├── data/
│   ├── preprocessing.py             # MRIPreprocessor, CTAPreprocessor classes
│   ├── dataset.py                   # AneurysmDataset, data loaders
│   └── select_dataset.py            # Build balanced 2000-series CSV
├── data_splits/
│   └── selected_dataset_2000.csv    # Curated dataset index
├── Preprocessed_images_2.5D/
│   ├── CTA/
│   │   ├── aneurysm/                # 500 .npy files
│   │   └── no_aneurysm/             # 500 .npy files
│   ├── MRA/
│   │   ├── aneurysm/                # 250 .npy files
│   │   └── no_aneurysm/             # 250 .npy files
│   └── MRI/
│       ├── aneurysm/                # 250 .npy files
│       └── no_aneurysm/             # 250 .npy files
├── outputs/
│   ├── multitask_*_kfold/           # Per-modality training outputs
│   │   └── fold_{k}/               # Models, plots, metrics per fold
│   ├── fusion_*_experiment/         # Fusion training outputs
│   │   └── best_fusion_model.pth   # Final deployed model
│   └── single_overlay/             # Single-image inference outputs
├── utils/
│   └── visualization.py             # Plotting helpers
│
├── preprocess_dataset.py            # DICOM → 2.5D .npy preprocessing
├── train_multitask.py               # Multi-task ResNet50 training (per modality)
├── train_fusion.py                  # Attention fusion model training
├── run_fusion_training.py           # Orchestrates fusion training
├── inference_fusion.py              # Batch inference with fusion model
├── infer_single_overlay.py          # Single-image inference + visual overlay
├── visualize_location_priors.py     # Batch heatmap generation
├── visualize_preprocessing.py       # Compare raw vs preprocessed
├── check_data_quality.py            # Dataset quality checks
└── check_fusion_ready.py            # Pre-flight checks before fusion training
```

### Training Sequence

```
Step 1:  python preprocess_dataset.py                    # DICOM → .npy
Step 2:  python train_multitask.py --modality CTA        # Train CTA model
Step 3:  python train_multitask.py --modality MRA        # Train MRA model
Step 4:  python train_multitask.py --modality MRI        # Train MRI model
Step 5:  python run_fusion_training.py                   # Train fusion model
Step 6:  python inference_fusion.py                      # Evaluate on test set
Step 7:  python infer_single_overlay.py --input ...      # Single-image demo
```

---

## Summary

The Aerux pipeline is a **multi-stage, multi-modal deep learning system** for intracranial aneurysm analysis:

1. **Preprocessing** converts raw DICOM volumes into standardized 2.5D representations with modality-specific image enhancements
2. **Multi-task learning** trains shared-backbone models that simultaneously detect, localize (13 arteries), and segment aneurysms
3. **Attention-based fusion** combines the three modality-specific models into a unified predictor
4. **Inference** produces detection results, location rankings, and visual heatmap overlays

For a **web application**, the existing `infer_single_overlay.py` logic serves as the direct backend — accepting an uploaded scan, preprocessing it, running the fusion model, and returning structured JSON with detection results, location probabilities, and overlay images. The model loads once at server startup and processes each request in under a second on GPU hardware.
