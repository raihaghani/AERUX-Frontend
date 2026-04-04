import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Inference Script for Fused Multi-Modal Model

This script loads the trained fusion model and generates predictions on test/inference data.

Usage:
    # Inference on default evaluation split (global test if use_global_split, else legacy fusion test)
    python inference_fusion.py --fusion_model path/to/best_fusion_model.pth
    
    # Inference on custom CSV
    python inference_fusion.py --fusion_model path/to/best_fusion_model.pth --input_csv path/to/data.csv
    
    # Generate submission file
    python inference_fusion.py --fusion_model path/to/best_fusion_model.pth --submission
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.config.config import Config
from train_multitask import MultiTaskResNet50, LOCATION_LABELS
from train_fusion import SimpleFusionModel, MultiModalDataset, multimodal_collate_fn


def load_fusion_model(checkpoint_path, device):
    """Load the trained fusion model"""
    print(f"Loading fusion model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model paths from checkpoint if available
    if 'config' in checkpoint and 'checkpoint_paths' in checkpoint['config']:
        modality_checkpoints = checkpoint['config']['checkpoint_paths']
    else:
        # Try to find them automatically
        print("WARNING: Model paths not in checkpoint, attempting auto-detection...")
        import glob
        modality_checkpoints = {}
        for modality in ['CTA', 'MRA', 'MRI']:
            pattern = f"E:/Education/Aerux_Final/outputs/multitask_{modality}_*/best_model_{modality}.pth"
            found = glob.glob(pattern)
            if found:
                found.sort(key=os.path.getmtime, reverse=True)
                modality_checkpoints[modality] = found[0]
                print(f"  Found {modality}: {modality_checkpoints[modality]}")
    
    # Load individual modality models
    modality_models = {}
    for modality, model_path in modality_checkpoints.items():
        print(f"Loading {modality} base model...")
        model = MultiTaskResNet50(num_detection_classes=2, num_location_classes=13, 
                                   input_channels=7, pretrained=False)
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        modality_models[modality] = model
    
    # Create fusion model
    fusion_strategy = checkpoint.get('fusion_strategy', 'weighted')
    fusion_model = SimpleFusionModel(modality_models, fusion_strategy=fusion_strategy)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model = fusion_model.to(device)
    fusion_model.eval()
    
    print("[OK] Fusion model loaded successfully\n")
    return fusion_model


def run_inference(model, dataloader, device):
    """Run inference on a dataset"""
    model.eval()
    
    all_series_uids = []
    all_det_probs = []
    all_det_preds = []
    all_loc_probs = []
    all_seg_preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc='Inference')):
            # Unpack batch
            if len(batch_data) == 5:
                inputs, det_labels, loc_labels, seg_masks, modality_indices = batch_data
            else:
                # Handle case where no labels are provided
                inputs, modality_indices = batch_data[0], batch_data[-1]
            
            # Skip empty batches
            batch_size = sum(len(indices) for indices in modality_indices.values())
            if batch_size == 0:
                continue
            
            # Move inputs to device
            inputs_device = {}
            for mod, vol in inputs.items():
                if vol is not None:
                    inputs_device[mod] = vol.to(device)
                else:
                    inputs_device[mod] = None
            
            # Forward pass
            det_logits, loc_logits, seg_pred = model(inputs_device, modality_indices)
            
            # Convert to probabilities
            det_probs = F.softmax(det_logits, dim=1).cpu().numpy()
            det_preds = torch.argmax(det_logits, dim=1).cpu().numpy()
            loc_probs = torch.sigmoid(loc_logits).cpu().numpy()
            seg_pred_np = torch.sigmoid(seg_pred).cpu().numpy()
            
            all_det_probs.append(det_probs)
            all_det_preds.extend(det_preds)
            all_loc_probs.append(loc_probs)
            all_seg_preds.append(seg_pred_np)
    
    # Concatenate results
    all_det_probs = np.vstack(all_det_probs)
    all_loc_probs = np.vstack(all_loc_probs)
    all_seg_preds = np.concatenate(all_seg_preds, axis=0)
    
    return {
        'detection_probs': all_det_probs,
        'detection_preds': np.array(all_det_preds),
        'location_probs': all_loc_probs,
        'segmentation_preds': all_seg_preds
    }


def create_submission_format(df, predictions, output_path):
    """Create competition submission file"""
    print("\nCreating submission file...")
    
    # Detection predictions (Aneurysm Present)
    df['Aneurysm_Present_Pred'] = predictions['detection_preds']
    df['Aneurysm_Present_Prob'] = predictions['detection_probs'][:, 1]
    
    # Location predictions (13 columns)
    for i, location in enumerate(LOCATION_LABELS):
        df[f'{location}_Prob'] = predictions['location_probs'][:, i]
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"[OK] Submission saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Inference with Fused Multi-Modal Model')
    parser.add_argument('--fusion_model', type=str, required=True,
                       help='Path to trained fusion model checkpoint')
    parser.add_argument('--input_csv', type=str, default=None,
                       help='Input CSV file (if not provided, uses test split from selected dataset)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions (default: same as model dir)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--submission', action='store_true',
                       help='Generate competition submission format')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.fusion_model):
        print(f"ERROR: Fusion model not found at {args.fusion_model}")
        return
    
    # Setup
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_fusion_model(args.fusion_model, device)
    
    # Prepare output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.fusion_model)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    if args.input_csv is not None:
        # Use provided CSV
        df = pd.read_csv(args.input_csv)
        print(f"Loaded {len(df)} samples from {args.input_csv}")
    else:
        from src.datasets.global_split_utils import load_evaluation_test_dataframe

        df = load_evaluation_test_dataframe(config)
        print(f"Using evaluation test split: {len(df)} samples")
        print(f"  Aneurysm present: {df['Aneurysm Present'].sum()}")
        print(f"  No aneurysm: {len(df) - df['Aneurysm Present'].sum()}")
    
    # Base directories
    base_dirs = {
        'CTA': config.data.preprocessed_cta_dir,
        'MRA': config.data.preprocessed_mra_dir,
        'MRI': config.data.preprocessed_mri_dir
    }
    
    # Create dataset
    train_csv_for_labels = args.input_csv if args.input_csv else config.data.train_csv
    inference_dataset = MultiModalDataset(
        df, base_dirs,
        train_csv_path=train_csv_for_labels,
        segmentation_dir=None,
        augment=False
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )
    
    # Run inference
    print("=" * 80)
    predictions = run_inference(model, inference_loader, device)
    print("=" * 80)
    
    # Save predictions
    print("\nSaving predictions...")
    
    # 1. Save raw predictions (numpy arrays)
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(predictions_path,
             detection_probs=predictions['detection_probs'],
             detection_preds=predictions['detection_preds'],
             location_probs=predictions['location_probs'],
             segmentation_preds=predictions['segmentation_preds'])
    print(f"[OK] Raw predictions saved to: {predictions_path}")
    
    # 2. Save human-readable CSV
    results_df = df.copy()
    results_df['Predicted_Aneurysm'] = predictions['detection_preds']
    results_df['Aneurysm_Probability'] = predictions['detection_probs'][:, 1]
    
    # Add location probabilities
    for i, location in enumerate(LOCATION_LABELS):
        results_df[f'{location}_Prob'] = predictions['location_probs'][:, i]
    
    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Predictions CSV saved to: {csv_path}")
    
    # 3. Save summary statistics
    summary = {
        'total_samples': len(df),
        'predicted_aneurysm': int(predictions['detection_preds'].sum()),
        'predicted_no_aneurysm': int(len(df) - predictions['detection_preds'].sum()),
        'mean_aneurysm_probability': float(predictions['detection_probs'][:, 1].mean()),
        'std_aneurysm_probability': float(predictions['detection_probs'][:, 1].std()),
    }
    
    # If ground truth available, compute metrics
    if 'Aneurysm Present' in df.columns:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        from train_multitask import compute_weighted_auc
        
        true_labels = df['Aneurysm Present'].values
        
        # Get location labels if available
        if all(loc in df.columns for loc in LOCATION_LABELS):
            loc_labels = df[LOCATION_LABELS].values
            
            weighted_auc, det_auc, loc_aucs = compute_weighted_auc(
                predictions['detection_probs'][:, 1],
                true_labels,
                predictions['location_probs'],
                loc_labels
            )
            
            summary['metrics'] = {
                'accuracy': float(accuracy_score(true_labels, predictions['detection_preds'])),
                'detection_auc': float(det_auc),
                'weighted_auc': float(weighted_auc),
                'f1_score': float(f1_score(true_labels, predictions['detection_preds'])),
                'location_aucs': {LOCATION_LABELS[i]: float(loc_aucs[i]) for i in range(13)}
            }
        else:
            summary['metrics'] = {
                'accuracy': float(accuracy_score(true_labels, predictions['detection_preds'])),
                'detection_auc': float(roc_auc_score(true_labels, predictions['detection_probs'][:, 1])),
                'f1_score': float(f1_score(true_labels, predictions['detection_preds']))
            }
    
    summary_path = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to: {summary_path}")
    
    # 4. Competition submission format (if requested)
    if args.submission:
        submission_path = os.path.join(args.output_dir, 'submission.csv')
        create_submission_format(df, predictions, submission_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Predicted aneurysm: {summary['predicted_aneurysm']} ({summary['predicted_aneurysm']/summary['total_samples']*100:.1f}%)")
    print(f"Predicted no aneurysm: {summary['predicted_no_aneurysm']} ({summary['predicted_no_aneurysm']/summary['total_samples']*100:.1f}%)")
    print(f"Mean aneurysm probability: {summary['mean_aneurysm_probability']:.4f}")
    
    if 'metrics' in summary:
        print("\nMETRICS (with ground truth):")
        print(f"  Accuracy: {summary['metrics']['accuracy']:.4f}")
        print(f"  Detection AUC: {summary['metrics']['detection_auc']:.4f}")
        if 'weighted_auc' in summary['metrics']:
            print(f"  Weighted AUC (Competition): {summary['metrics']['weighted_auc']:.4f}")
        print(f"  F1 Score: {summary['metrics']['f1_score']:.4f}")
    
    print("=" * 80)
    print(f"\n[OK] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
