import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Test-set inference with per-sample heatmap overlays and aggregate metrics.

When config.data.use_global_split is True (recommended):
    Evaluates on global_test_series.csv only — same UIDs never used in multitask or fusion training.
    Create splits first: python create_global_split.py

Legacy mode (use_global_split False):
    Reproduces the internal 70/15/15 fusion test split from the full filtered pool.

Outputs (under --out_dir):
    samples/              per-sample overlay, heatmap, mask, input, prediction.json
    predictions.csv       all predictions + ground truth
    metrics.json          aggregate metrics (accuracy, AUC, F1, per-location AUC…)
    confusion_matrix.png
    roc_curve.png
    location_aucs.png

Usage:
    python infer_test_set.py --fusion_model path/to/best_fusion_model.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

sns.set_style("whitegrid")

sys.path.insert(0, str(Path(__file__).parent))
from src.config.config import Config
from inference_fusion import load_fusion_model
from train_multitask import LOCATION_LABELS, compute_weighted_auc
from infer_single_overlay import (
    LOCATION_ANCHORS,
    gaussian_map,
    build_location_prior_heatmap,
    colorize_and_overlay,
    save_image,
)
from preprocess_dataset import normalize_modality
from src.datasets.global_split_utils import load_evaluation_test_dataframe


def load_npy_7ch(npy_path: str):
    """Load a preprocessed .npy and return (vol_7ch, display_gray)."""
    arr = np.load(npy_path).astype(np.float32)
    if arr.shape[0] == 7:
        vol = arr
    elif arr.shape[-1] == 7:
        vol = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected .npy shape: {arr.shape}")

    center = vol[3]
    cmin, cmax = float(center.min()), float(center.max())
    if cmax > cmin:
        display = ((center - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)
    else:
        display = np.zeros_like(center, dtype=np.uint8)
    return vol, display


def infer_single(model, vol_7ch, modality, device):
    """Run fusion model on a single 7-channel volume and return probs."""
    x = torch.from_numpy(vol_7ch).unsqueeze(0).to(device)

    inputs = {"CTA": None, "MRA": None, "MRI": None}
    modality_indices = {"CTA": [], "MRA": [], "MRI": []}
    inputs[modality] = x
    modality_indices[modality] = [0]

    with torch.no_grad():
        det_logits, loc_logits, _ = model(inputs, modality_indices)
        det_probs = F.softmax(det_logits, dim=1).cpu().numpy()[0]
        loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]

    return det_probs, loc_probs


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Aneurysm", "Aneurysm"],
                yticklabels=["No Aneurysm", "Aneurysm"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Set — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(true_labels, pred_probs, det_auc, save_path):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {det_auc:.4f}")
    plt.plot([0, 1], [0, 1], "r--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test Set — ROC Curve (Detection)")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_location_aucs(loc_aucs, save_path):
    x = np.arange(len(LOCATION_LABELS))
    colors = ["green" if a >= 0.8 else "steelblue" if a >= 0.6 else "orange" if a >= 0.5 else "red" for a in loc_aucs]
    plt.figure(figsize=(14, 7))
    plt.bar(x, loc_aucs, color=colors, alpha=0.8)
    plt.xticks(x, LOCATION_LABELS, rotation=45, ha="right", fontsize=9)
    plt.ylabel("AUC")
    plt.title("Test Set — AUC per Anatomical Location")
    plt.axhline(y=0.5, color="r", linestyle="--", lw=1.5, label="Random (0.5)")
    plt.axhline(y=0.8, color="g", linestyle="--", lw=1.5, alpha=0.5, label="Good (0.8)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run fusion inference on the full test set with overlays")
    parser.add_argument("--fusion_model", type=Path, required=True, help="Path to best_fusion_model.pth")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: outputs/test_set_inference_<timestamp>)")
    parser.add_argument("--sigma", type=float, default=18.0, help="Gaussian spread for heatmap")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--threshold", type=float, default=0.45, help="Binary mask threshold")
    parser.add_argument("--save_overlays", action="store_true", default=True, help="Save per-sample overlay images")
    parser.add_argument("--no_overlays", action="store_true", help="Skip per-sample overlay generation (metrics only)")
    args = parser.parse_args()

    if args.no_overlays:
        args.save_overlays = False

    if not args.fusion_model.exists():
        raise FileNotFoundError(f"Fusion model not found: {args.fusion_model}")

    config = Config()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(config.data.output_root) / f"test_set_inference_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "samples"
    if args.save_overlays:
        samples_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Same evaluation CSV logic as inference_fusion.py default branch
    print("\nLoading evaluation split …")
    test_df = load_evaluation_test_dataframe(config)
    if getattr(config.data, "use_global_split", False):
        print("Using GLOBAL held-out test (global_test_series.csv).")
    else:
        print("Using LEGACY internal fusion test split (may overlap multitask training data).")

    base_dirs = {
        "CTA": config.data.preprocessed_cta_dir,
        "MRA": config.data.preprocessed_mra_dir,
        "MRI": config.data.preprocessed_mri_dir,
    }
    test_df = test_df.copy()
    test_df["Modality"] = test_df["Modality"].apply(normalize_modality)

    print(f"Test set: {len(test_df)} samples")
    print(f"  Aneurysm:    {int(test_df['Aneurysm Present'].sum())}")
    print(f"  No aneurysm: {int(len(test_df) - test_df['Aneurysm Present'].sum())}")
    for mod in ("CTA", "MRA", "MRI"):
        print(f"  {mod}: {len(test_df[test_df['Modality'] == mod])}")

    # Load fusion model
    model = load_fusion_model(str(args.fusion_model), device)

    # Inference loop
    all_det_probs = []
    all_loc_probs = []
    all_det_preds = []
    all_true_labels = []
    records = []

    print(f"\nRunning inference on {len(test_df)} test samples …")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        series_uid = row["SeriesInstanceUID"]
        modality = row["Modality"]
        true_label = int(row["Aneurysm Present"])
        label_name = "aneurysm" if true_label == 1 else "no_aneurysm"
        npy_path = os.path.join(base_dirs[modality], label_name, f"{series_uid}.npy")

        vol_7ch, display_gray = load_npy_7ch(npy_path)
        det_probs, loc_probs = infer_single(model, vol_7ch, modality, device)

        det_pred = int(np.argmax(det_probs))
        det_prob = float(det_probs[1])

        all_det_probs.append(det_prob)
        all_loc_probs.append(loc_probs)
        all_det_preds.append(det_pred)
        all_true_labels.append(true_label)

        rec = {
            "SeriesInstanceUID": series_uid,
            "Modality": modality,
            "TrueLabel": true_label,
            "PredLabel": det_pred,
            "AneurysmProb": det_prob,
        }
        for i, label in enumerate(LOCATION_LABELS):
            rec[f"{label}_Prob"] = float(loc_probs[i])
        records.append(rec)

        # Per-sample overlay images
        if args.save_overlays:
            sample_dir = samples_dir / f"{series_uid}"
            sample_dir.mkdir(exist_ok=True)

            save_image(display_gray, sample_dir / "input_gray.png")

            if det_pred == 1:
                heat, loc_weights = build_location_prior_heatmap(
                    det_prob, loc_probs,
                    display_gray.shape[0], display_gray.shape[1],
                    args.sigma, use_detection_weight=True,
                )
                heat_uint8, binary_mask, heat_color, overlay = colorize_and_overlay(
                    display_gray, heat, args.alpha, args.threshold,
                )

                save_image(heat_color, sample_dir / "heatmap.png")
                save_image(binary_mask, sample_dir / "highlight_mask.png")
                save_image(overlay, sample_dir / "overlay.png")

                top_locs = sorted(loc_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
            else:
                loc_weights = {LOCATION_LABELS[i]: float(det_prob * loc_probs[i]) for i in range(len(LOCATION_LABELS))}
                top_locs = sorted(loc_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]

            pred_json = {
                "series_uid": series_uid,
                "modality": modality,
                "true_label": true_label,
                "detection_pred": det_pred,
                "aneurysm_prob": det_prob,
                "top_3_locations": [{"label": l, "score": float(s)} for l, s in top_locs],
                "correct": det_pred == true_label,
            }
            with open(sample_dir / "prediction.json", "w") as f:
                json.dump(pred_json, f, indent=2)

    # Aggregate metrics
    all_det_probs = np.array(all_det_probs)
    all_det_preds = np.array(all_det_preds)
    all_true_labels = np.array(all_true_labels)
    all_loc_probs = np.vstack(all_loc_probs)

    # Build location ground-truth matrix from train.csv
    train_csv_df = pd.read_csv(config.data.train_csv)
    uid_to_loc = {}
    for _, r in train_csv_df.iterrows():
        vec = np.zeros(13, dtype=np.float32)
        for i, label in enumerate(LOCATION_LABELS):
            if label in r and pd.notna(r[label]) and r[label] == 1:
                vec[i] = 1.0
        uid_to_loc[r["SeriesInstanceUID"]] = vec

    all_loc_labels = np.stack([
        uid_to_loc.get(row["SeriesInstanceUID"], np.zeros(13, dtype=np.float32))
        for _, row in test_df.iterrows()
    ])

    accuracy = accuracy_score(all_true_labels, all_det_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_det_preds, average="binary", zero_division=0,
    )
    det_auc = roc_auc_score(all_true_labels, all_det_probs)
    weighted_auc, _, loc_aucs = compute_weighted_auc(
        all_det_probs, all_true_labels, all_loc_probs, all_loc_labels,
    )
    cm = confusion_matrix(all_true_labels, all_det_preds)

    metrics = {
        "evaluation_split": "global_held_out"
        if getattr(config.data, "use_global_split", False)
        else "legacy_internal_fusion_test",
        "test_samples": int(len(test_df)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "detection_auc": float(det_auc),
        "weighted_auc": float(weighted_auc),
        "confusion_matrix": cm.tolist(),
        "location_aucs": {LOCATION_LABELS[i]: float(loc_aucs[i]) for i in range(13)},
        "correct": int((all_det_preds == all_true_labels).sum()),
        "incorrect": int((all_det_preds != all_true_labels).sum()),
    }

    # Save outputs
    pred_df = pd.DataFrame(records)
    pred_df.to_csv(args.out_dir / "predictions.csv", index=False)

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, args.out_dir / "confusion_matrix.png")
    plot_roc(all_true_labels, all_det_probs, det_auc, args.out_dir / "roc_curve.png")
    plot_location_aucs(loc_aucs, args.out_dir / "location_aucs.png")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SET INFERENCE RESULTS")
    print("=" * 80)
    print(f"  Samples:       {metrics['test_samples']}")
    print(f"  Correct:       {metrics['correct']}  |  Incorrect: {metrics['incorrect']}")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Detection AUC: {det_auc:.4f}")
    print(f"  Weighted AUC:  {weighted_auc:.4f}")
    print()
    print("  Per-location AUCs:")
    for i, label in enumerate(LOCATION_LABELS):
        print(f"    {label:50s}  {loc_aucs[i]:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print(f"All outputs saved to: {args.out_dir}")
    if args.save_overlays:
        print(f"  Per-sample overlays: {samples_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
