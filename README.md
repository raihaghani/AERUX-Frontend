# AERUX: Intracranial Aneurysm Detection System

AERUX is a comprehensive, AI-powered diagnostic web application for the automatic detection, localization, and segmentation of intracranial aneurysms from 3D medical scans.

This repository holds the entire consolidated stack:

1. **Next.js Frontend (`aerux-web/`)**: React application, UI Dashboard, PDF Reports.
2. **FastAPI Backend (`Backend/`)**: PyTorch inference server, ML model loading, DICOM & NIfTI preprocessing.

---

## ⚡ TL;DR: Quickstart - How to Run Both Servers

To run the application locally, you need to start **both** the backend and the frontend in two separate terminal windows.

**Terminal 1 (Backend - FastAPI):**

```bash
cd Backend
venv\Scripts\activate      # Or `source venv/bin/activate` on Mac/Linux
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend - Next.js):**

```bash
cd aerux-web
npm run dev
```

Finally, open your browser and navigate to **http://localhost:3000**

---

## 📅 Recent Updates (April 2026)
- **AI Chatbot Hardening**: Enforced strict medical bounds on the chatbot API. It will now firmly refuse out-of-scope non-medical queries and focus entirely on aneurysm evaluation, the AERUX workflow, and neurovascular anatomy.
- **Backend Sync & Cleanup**: Cleaned up the Python package structure and fixed faulty imports (`ModuleNotFoundError`) in the `src/` modules, ensuring flawless synchronization with upstream inference scripts.
- **Config Hotfix**: Addressed widespread UTF-16 decoding errors (`UnicodeDecodeError`) on Windows affecting both `.env` and `.env.local`, stabilizing FastAPI boot times and preserving Node context loading.

---

## 🏗️ Architecture & Integration Flow

The frontend and backend run as separate local servers that securely talk to each other server-to-server.

1. **User Upload**: The user drops a scan (`.dcm`, `.nii`, `.nii.gz`, `.png`, `.npy`) into the Next.js frontend on port 3000.
2. **Next.js Proxy (`POST /api/upload`)**: Next.js receives the file and invisibly forwards it to the FastAPI backend running on port 8000 via a server-side `fetch`. The browser never hits port 8000 directly to avoid CORS issues.
3. **FastAPI Processing (`POST /predict`)**:
   - `preprocess_upload()` standardises the input (e.g. extracts 2.5D slices from DICOM series, normalizes, resizes).
   - The PyTorch ResNet50 fusion model predicts aneurysm probability and localises the 13 distinct arteries using a threaded, non-blocking executor.
   - Heatmap overlays and diagnosis JSONs are saved to the backend disk (`Backend/inference_outputs/`).
4. **Next.js Image Proxy (`GET /api/images/[id]/[filename]`)**: The frontend React dashboard requests the heatmaps via Next.js, which streams the image bytes from FastAPI back to the browser safely.

---

## ⚙️ How to Set Up on a New Device

### Requirements

- **Node.js** v18+ (for Next.js)
- **Python** 3.10+ (for FastAPI and PyTorch)
- **Git** (to clone the repo)
- CUDA-compatible GPU (Highly recommended for fast inference)

### 1. Clone the Repository

```bash
git clone https://github.com/raihaghani/AERUX-Frontend.git
cd AERUX-Frontend
```

### 2. Set Up the Python Backend

First, ensure you have the trained model weights.
Download `best_fusion_model.pth` and place it somewhere accessible (e.g., `C:/models/best_fusion_model.pth`).

```bash
cd Backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version as needed
pip install fastapi uvicorn python-multipart python-dotenv numpy opencv-python SimpleITK pandas

# Create your .env file
echo FUSION_MODEL_PATH=C:/models/best_fusion_model.pth > .env
echo UPLOAD_DIR=./uploads >> .env
echo OUTPUT_DIR=./inference_outputs >> .env
echo MAX_UPLOAD_MB=200 >> .env
echo RESULT_TTL_HOURS=24 >> .env
```

> **Note:** Update `FUSION_MODEL_PATH` in `.env` to precisely point to where you saved the `.pth` model file on your new device.

### 3. Set Up the Next.js Frontend

```bash
# From the root repository folder, navigate to the web app
cd aerux-web

# Install all Node dependencies
npm install

# Create the environment file linking Next.js to the FastAPI backend
echo FASTAPI_BASE_URL=http://localhost:8000 > .env.local
```

---

## 🚀 How to Run the System

You must start **both** servers for the application to work. It is easiest to open two separate terminal windows.

### Terminal 1: Start the Backend (FastAPI)

```bash
cd AERUX-Frontend/Backend

# 1. Activate the environment (if not already active)
venv\Scripts\activate

# 2. Start Uvicorn
uvicorn main:app --reload --port 8000
```

_Wait until you see `[startup] Model ready.`_

### Terminal 2: Start the Frontend (Next.js)

```bash
cd AERUX-Frontend/aerux-web

# Start the Next.js development server
npm run dev
```

### Accessing the App

Open your web browser and navigate to:
**👉 http://localhost:3000**

You can now upload scans, view probabilities on the Visuals Dashboard, and download generated PDF reports without running any isolated Python scripts!

---

## 🧹 Maintenance and Artifacts

- **Uploads & Outputs:** The FastAPI server automatically deletes processed scans and images from the `Backend/uploads` and `Backend/inference_outputs` directories after 24 hours (`RESULT_TTL_HOURS`).
- **Logs:** If an upload fails, check Terminal 1 (FastAPI output) to view Python/PyTorch traceback logs.
