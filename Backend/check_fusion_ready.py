"""
Pre-flight Check for Late Fusion Training

Run this script before training to verify all requirements are met.
"""

import os
import pandas as pd
import glob

print("=" * 80)
print("LATE FUSION TRAINING - PRE-FLIGHT CHECK")
print("=" * 80)

all_checks_passed = True

# Check 1: Selected dataset exists
print("\n[1/6] Checking selected dataset...")
selected_dataset_path = "E:/Education/Aerux_Final/data_splits/selected_dataset_2000.csv"
if os.path.exists(selected_dataset_path):
    df = pd.read_csv(selected_dataset_path)
    print(f"  ✓ Found selected_dataset_2000.csv")
    print(f"  ✓ Total samples: {len(df)}")
    print(f"    - CTA: {len(df[df['Modality'] == 'CTA'])}")
    print(f"    - MRA: {len(df[df['Modality'] == 'MRA'])}")
    print(f"    - MRI: {len(df[df['Modality'].str.contains('MRI')])}")
else:
    print(f"  ✗ ERROR: Selected dataset not found at {selected_dataset_path}")
    all_checks_passed = False

# Check 2: Preprocessed data directories exist
print("\n[2/6] Checking preprocessed data directories...")
preprocessed_base = "E:/Education/Aerux_Final/Preprocessed_images_2.5D"
modalities = ['CTA', 'MRA', 'MRI']

for modality in modalities:
    modality_dir = os.path.join(preprocessed_base, modality)
    if os.path.exists(modality_dir):
        # Count aneurysm and no_aneurysm files
        aneurysm_dir = os.path.join(modality_dir, 'aneurysm')
        no_aneurysm_dir = os.path.join(modality_dir, 'no_aneurysm')
        
        aneurysm_count = len(glob.glob(os.path.join(aneurysm_dir, '*.npy'))) if os.path.exists(aneurysm_dir) else 0
        no_aneurysm_count = len(glob.glob(os.path.join(no_aneurysm_dir, '*.npy'))) if os.path.exists(no_aneurysm_dir) else 0
        
        print(f"  ✓ {modality}: {aneurysm_count + no_aneurysm_count} files ({aneurysm_count} aneurysm, {no_aneurysm_count} no_aneurysm)")
    else:
        print(f"  ✗ ERROR: {modality} directory not found at {modality_dir}")
        all_checks_passed = False

# Check 3: Trained model checkpoints exist
print("\n[3/6] Checking trained model checkpoints...")
for modality in modalities:
    pattern = f"E:/Education/Aerux_Final/outputs/multitask_{modality}_*/best_model_{modality}.pth"
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        # Sort by modification time, most recent first
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        latest = checkpoints[0]
        mod_time = os.path.getmtime(latest)
        from datetime import datetime
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  ✓ {modality} model found: {os.path.basename(os.path.dirname(latest))}")
        print(f"    Last modified: {mod_date}")
    else:
        print(f"  ✗ ERROR: No {modality} checkpoint found")
        print(f"    Expected pattern: {pattern}")
        all_checks_passed = False

# Check 4: CUDA availability
print("\n[4/6] Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠ WARNING: CUDA not available, will use CPU (slower)")
except ImportError:
    print(f"  ✗ ERROR: PyTorch not installed")
    all_checks_passed = False

# Check 5: Required Python packages
print("\n[5/6] Checking required packages...")
required_packages = [
    'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 
    'seaborn', 'sklearn', 'tqdm'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (missing)")
        missing_packages.append(package)
        all_checks_passed = False

if missing_packages:
    print(f"\n  Install missing packages with:")
    print(f"  pip install {' '.join(missing_packages)}")

# Check 6: Output directory writable
print("\n[6/6] Checking output directory...")
output_root = "E:/Education/Aerux_Final/outputs"
if os.path.exists(output_root):
    print(f"  ✓ Output directory exists: {output_root}")
    # Try to create a test directory
    test_dir = os.path.join(output_root, 'test_write')
    try:
        os.makedirs(test_dir, exist_ok=True)
        os.rmdir(test_dir)
        print(f"  ✓ Output directory is writable")
    except:
        print(f"  ✗ ERROR: Cannot write to output directory")
        all_checks_passed = False
else:
    print(f"  ⚠ WARNING: Output directory does not exist, will be created")

# Final summary
print("\n" + "=" * 80)
if all_checks_passed:
    print("✓ ALL CHECKS PASSED - Ready to start late fusion training!")
    print("\nRun: python run_fusion_training.py")
else:
    print("✗ SOME CHECKS FAILED - Please fix the errors above before training")
print("=" * 80)
