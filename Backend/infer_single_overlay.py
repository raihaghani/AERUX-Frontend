"""
Single-image inference + weak localization overlay for fusion model.

This script runs the trained fusion model on ONE input image and creates:
1) prediction JSON (binary + 13 location probabilities)
2) location-prior heatmap
3) heatmap overlay on the input image

NOTE:
- Best input is preprocessed 2.5D `.npy` used by your training pipeline
  (shape `(H, W, 7)` or `(7, H, W)`, typically 256x256).
- If you pass `.png/.jpg/.jpeg`, the script converts to grayscale, resizes
  to 256x256, and tiles to 7 channels. This is for convenience only and may
  be less reliable than true preprocessed `.npy` inputs.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inference_fusion import load_fusion_model
from constants import LOCATION_LABELS


LOCATION_ANCHORS = {
    "Left Infraclinoid Internal Carotid Artery": (0.40, 0.62),
    "Right Infraclinoid Internal Carotid Artery": (0.60, 0.62),
    "Left Supraclinoid Internal Carotid Artery": (0.42, 0.53),
    "Right Supraclinoid Internal Carotid Artery": (0.58, 0.53),
    "Left Middle Cerebral Artery": (0.30, 0.46),
    "Right Middle Cerebral Artery": (0.70, 0.46),
    "Anterior Communicating Artery": (0.50, 0.40),
    "Left Anterior Cerebral Artery": (0.43, 0.36),
    "Right Anterior Cerebral Artery": (0.57, 0.36),
    "Left Posterior Communicating Artery": (0.43, 0.57),
    "Right Posterior Communicating Artery": (0.57, 0.57),
    "Basilar Tip": (0.50, 0.55),
    "Other Posterior Circulation": (0.50, 0.70),
}


def load_input_as_7ch(input_path: Path, target_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Load input and return (model_input_7ch, display_gray_image).

    model_input_7ch: float32 array shape (7, H, W)
    display_gray_image: uint8 array shape (H, W)
    """
    suffix = input_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(input_path).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D npy array, got shape {arr.shape}")

        if arr.shape[0] == 7:
            vol = arr
        elif arr.shape[-1] == 7:
            vol = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Expected 7 channels in first or last axis, got shape {arr.shape}")

        h, w = vol.shape[1], vol.shape[2]
        if (h, w) != (target_size, target_size):
            try:
                import cv2

                resized = []
                for c in range(7):
                    resized.append(cv2.resize(vol[c], (target_size, target_size), interpolation=cv2.INTER_LINEAR))
                vol = np.stack(resized, axis=0).astype(np.float32)
            except Exception:
                raise ValueError(
                    f"Input size {(h, w)} != {(target_size, target_size)} and OpenCV unavailable for resize."
                )

        center = vol[3]
        center_min, center_max = float(center.min()), float(center.max())
        if center_max > center_min:
            display = ((center - center_min) / (center_max - center_min) * 255.0).astype(np.uint8)
        else:
            display = np.zeros((target_size, target_size), dtype=np.uint8)

        return vol.astype(np.float32), display

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        try:
            import cv2

            img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            display = img.astype(np.uint8)
            x = display.astype(np.float32) / 255.0
            vol = np.stack([x] * 7, axis=0)
            print("WARNING: Using 2D image tiled into 7 channels. Prefer preprocessed .npy for best results.")
            return vol.astype(np.float32), display
        except Exception:
            from PIL import Image

            img = Image.open(input_path).convert("L").resize((target_size, target_size))
            display = np.array(img, dtype=np.uint8)
            x = display.astype(np.float32) / 255.0
            vol = np.stack([x] * 7, axis=0)
            print("WARNING: Using 2D image tiled into 7 channels. Prefer preprocessed .npy for best results.")
            return vol.astype(np.float32), display

    raise ValueError("Supported inputs: .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff")


def gaussian_map(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)


def build_location_prior_heatmap(
    det_prob: float,
    location_probs: np.ndarray,
    h: int,
    w: int,
    sigma: float,
    use_detection_weight: bool = True,
) -> tuple[np.ndarray, dict]:
    if location_probs.shape[0] != len(LOCATION_LABELS):
        raise ValueError(f"Expected {len(LOCATION_LABELS)} location probs, got {location_probs.shape}")

    det_weight = max(0.0, min(1.0, float(det_prob))) if use_detection_weight else 1.0

    heat = np.zeros((h, w), dtype=np.float32)
    weights = {}

    for i, label in enumerate(LOCATION_LABELS):
        p = max(0.0, min(1.0, float(location_probs[i])))
        w_loc = p * det_weight
        weights[label] = w_loc

        nx, ny = LOCATION_ANCHORS[label]
        cx = nx * (w - 1)
        cy = ny * (h - 1)
        heat += w_loc * gaussian_map(h, w, cx, cy, sigma)

    return heat, weights


def colorize_and_overlay(gray_img: np.ndarray, heat: np.ndarray, alpha: float, threshold: float):
    if heat.max() > 0:
        heat_norm = (heat / heat.max()).astype(np.float32)
    else:
        heat_norm = np.zeros_like(heat, dtype=np.float32)

    heat_uint8 = (np.clip(heat_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    binary_mask = (heat_norm >= threshold).astype(np.uint8) * 255

    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)

    try:
        import cv2

        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(gray_rgb, 1.0 - alpha, heat_color, alpha, 0)
        return heat_uint8, binary_mask, heat_color, overlay
    except Exception:
        heat_color = np.zeros_like(gray_rgb)
        heat_color[..., 0] = heat_uint8
        overlay = np.clip((1.0 - alpha) * gray_rgb + alpha * heat_color, 0, 255).astype(np.uint8)
        return heat_uint8, binary_mask, heat_color, overlay


def save_image(img: np.ndarray, out_path: Path):
    try:
        import cv2

        cv2.imwrite(str(out_path), img)
        return "opencv"
    except Exception:
        from PIL import Image

        Image.fromarray(img).save(out_path)
        return "pillow"


def main():
    parser = argparse.ArgumentParser(description="Single-image fusion inference with heatmap overlay")
    parser.add_argument("--fusion_model", type=Path, required=True, help="Path to best_fusion_model.pth")
    parser.add_argument("--input", type=Path, required=True, help="Input image path (.npy preferred)")
    parser.add_argument("--modality", type=str, required=True, choices=["CTA", "MRA", "MRI"], help="Input modality")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/single_overlay"), help="Output directory")
    parser.add_argument("--sigma", type=float, default=18.0, help="Gaussian spread in pixels")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    parser.add_argument("--threshold", type=float, default=0.45, help="Threshold on normalized heatmap for binary highlight")
    parser.add_argument("--no_detection_weight", action="store_true", help="Ignore binary probability as multiplier")

    args = parser.parse_args()

    if not args.fusion_model.exists():
        raise FileNotFoundError(f"Fusion model not found: {args.fusion_model}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_fusion_model(str(args.fusion_model), device)

    vol_7ch, display_gray = load_input_as_7ch(args.input)
    x = torch.from_numpy(vol_7ch).unsqueeze(0).to(device)  # (1, 7, H, W)

    inputs = {"CTA": None, "MRA": None, "MRI": None}
    modality_indices = {"CTA": [], "MRA": [], "MRI": []}
    inputs[args.modality] = x
    modality_indices[args.modality] = [0]

    with torch.no_grad():
        det_logits, loc_logits, _ = model(inputs, modality_indices)
        det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]
        loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]

    det_pred = int(np.argmax(det_probs))
    det_prob = float(det_probs[1])

    heat, loc_weights = build_location_prior_heatmap(
        det_prob=det_prob,
        location_probs=loc_probs,
        h=display_gray.shape[0],
        w=display_gray.shape[1],
        sigma=args.sigma,
        use_detection_weight=not args.no_detection_weight,
    )

    heat_uint8, binary_mask, heat_color, overlay = colorize_and_overlay(
        gray_img=display_gray,
        heat=heat,
        alpha=args.alpha,
        threshold=args.threshold,
    )

    top_locations = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)[:3]

    result = {
        "input_path": str(args.input),
        "modality": args.modality,
        "detection_prediction": det_pred,
        "detection_probabilities": {
            "no_aneurysm": float(det_probs[0]),
            "aneurysm": float(det_probs[1]),
        },
        "top_3_locations": [
            {"label": label, "score": float(score)} for label, score in top_locations
        ],
        "all_location_probabilities": {
            LOCATION_LABELS[i]: float(loc_probs[i]) for i in range(len(LOCATION_LABELS))
        },
        "overlay_note": "Weak localization from location probabilities (not true segmentation).",
    }

    json_path = args.out_dir / "prediction.json"
    gray_path = args.out_dir / "input_gray.png"
    heat_path = args.out_dir / "heatmap.png"
    bin_path = args.out_dir / "highlight_mask.png"
    ovl_path = args.out_dir / "overlay.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    writer = save_image(display_gray, gray_path)
    writer = save_image(heat_color, heat_path)
    writer = save_image(binary_mask, bin_path)
    writer = save_image(overlay, ovl_path)

    print("\nInference complete")
    print(f"Detection prediction: {det_pred} (aneurysm prob={det_prob:.4f})")
    print("Top locations:")
    for label, score in top_locations:
        print(f"  - {label}: {score:.4f}")
    print(f"\nSaved to: {args.out_dir}")
    print(f"Image writer: {writer}")


if __name__ == "__main__":
    main()
