import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Script to run late fusion training with your 3 trained models

This script trains calibration layers on top of frozen CTA, MRA, and MRI models
to learn optimal fusion weights for combining predictions.

Usage:
    python run_fusion_training.py
"""

import os
import glob

# Find the latest trained model checkpoints
def find_latest_checkpoint(modality):
    """Find the most recent checkpoint for a given modality"""
    pattern = f"E:/Education/Aerux_Final/outputs/multitask_DenseNet_{modality}_*_kfold/fold_4/best_model_DenseNet_{modality}_fold4.pth"
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"ERROR: No {modality} checkpoint found matching pattern: {pattern}")
        return None
    
    # Sort by modification time, most recent first
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

# Find all model checkpoints
print("=" * 80)
print("SEARCHING FOR TRAINED MODEL CHECKPOINTS")
print("=" * 80)

cta_checkpoint = find_latest_checkpoint("CTA")
mra_checkpoint = find_latest_checkpoint("MRA")
mri_checkpoint = find_latest_checkpoint("MRI")

if not all([cta_checkpoint, mra_checkpoint, mri_checkpoint]):
    print("\nERROR: Could not find all required model checkpoints!")
    print("\nPlease ensure you have trained models for CTA, MRA, and MRI.")
    print("Expected locations:")
    print("  E:/Education/Aerux_Final/outputs/multitask_CTA_*_kfold/fold_4/best_model_CTA_fold4.pth")
    print("  E:/Education/Aerux_Final/outputs/multitask_MRA_*_kfold/fold_4/best_model_MRA_fold4.pth")
    print("  E:/Education/Aerux_Final/outputs/multitask_MRI_*_kfold/fold_4/best_model_MRI_fold4.pth")
    exit(1)

print(f"\n[OK] Found CTA model: {cta_checkpoint}")
print(f"[OK] Found MRA model: {mra_checkpoint}")
print(f"[OK] Found MRI model: {mri_checkpoint}")

# Build the command
command = f"""python train_fusion_Densenet.py \
    --cta_model "{cta_checkpoint}" \
    --mra_model "{mra_checkpoint}" \
    --mri_model "{mri_checkpoint}" \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4"""

print("\n" + "=" * 80)
print("RUNNING LATE FUSION TRAINING")
print("=" * 80)
print("\nCommand:")
print(command)
print("\n" + "=" * 80)

# Execute the command
os.system(command)
