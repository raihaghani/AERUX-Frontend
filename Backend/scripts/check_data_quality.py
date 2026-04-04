import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Diagnostic script to check segmentation masks and location labels
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config.config import Config

print("=" * 80)
print("DATA DIAGNOSTICS FOR FUSION TRAINING")
print("=" * 80)

config = Config()
dataset_path = config.data.train_csv
df = pd.read_csv(dataset_path)

print(f"\n1. TRAIN CSV OVERVIEW ({dataset_path})")
print(f"   Total samples: {len(df)}")
print(f"   Aneurysm cases: {df['Aneurysm Present'].sum()}")
print(f"   No aneurysm cases: {len(df) - df['Aneurysm Present'].sum()}")

# 2. Check location labels
location_cols = [
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
    "Other Posterior Circulation"
]

print(f"\n2. LOCATION LABELS ANALYSIS")
aneurysm_cases = df[df['Aneurysm Present'] == 1]
print(f"   Aneurysm cases with location labels: {aneurysm_cases[location_cols].any(axis=1).sum()}")

print(f"\n   Location distribution (top 5):")
location_counts = {}
for col in location_cols:
    count = df[col].sum()
    if count > 0:
        location_counts[col] = count

sorted_locs = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
for loc, count in sorted_locs[:5]:
    print(f"   - {loc}: {count}")

if len(sorted_locs) > 5:
    print(f"   ... and {len(sorted_locs) - 5} more locations")

# 3. Check segmentation masks
seg_dir = "D:/FYP_Data/rsna-intracranial-aneurysm-detection/segmentations"
print(f"\n3. SEGMENTATION MASKS CHECK")
print(f"   Segmentation directory: {seg_dir}")

if os.path.exists(seg_dir):
    mask_files = [f for f in os.listdir(seg_dir) if f.endswith('.npy')]
    print(f"   Total mask files available: {len(mask_files)}")
    
    # Check how many selected dataset cases have masks
    selected_with_masks = 0
    for series_uid in df[df['Aneurysm Present'] == 1]['SeriesInstanceUID']:
        mask_path = os.path.join(seg_dir, f"{series_uid}.npy")
        if os.path.exists(mask_path):
            selected_with_masks += 1
    
    print(f"   Selected aneurysm cases with masks: {selected_with_masks} / {df['Aneurysm Present'].sum()}")
    print(f"   Coverage: {selected_with_masks / df['Aneurysm Present'].sum() * 100:.1f}%")
else:
    print(f"    Segmentation directory NOT FOUND")
    print(f"   Segmentation training will use zero masks")

# 4. Check preprocessed data availability
print(f"\n4. PREPROCESSED DATA CHECK")
base_dirs = {
    'CTA': 'E:/Education/Aerux_Final/Preprocessed_images_2.5D/CTA',
    'MRA': 'E:/Education/Aerux_Final/Preprocessed_images_2.5D/MRA',
    'MRI': 'E:/Education/Aerux_Final/Preprocessed_images_2.5D/MRI'
}

for modality, base_dir in base_dirs.items():
    modality_df = df[df['Modality'].str.contains(modality)]
    aneurysm_dir = os.path.join(base_dir, 'aneurysm')
    no_aneurysm_dir = os.path.join(base_dir, 'no_aneurysm')
    
    found = 0
    missing = []
    
    for idx, row in modality_df.iterrows():
        series_uid = row['SeriesInstanceUID']
        label = 'aneurysm' if row['Aneurysm Present'] == 1 else 'no_aneurysm'
        file_path = os.path.join(base_dir, label, f"{series_uid}.npy")
        
        if os.path.exists(file_path):
            found += 1
        else:
            if len(missing) < 5:  # Store first 5 missing
                missing.append(series_uid[:30])
    
    print(f"   {modality}: {found}/{len(modality_df)} files found ({found/len(modality_df)*100:.1f}%)")
    if missing:
        print(f"      Missing files (first {len(missing)}): {missing[0]}...")

# 5. Recommendations
print(f"\n5. RECOMMENDATIONS")
print(f"   ✓ Location labels: AVAILABLE ({len([c for c in location_counts.values() if c > 0])} locations used)")
print(f"   ✓ Calibration layers: Will learn from location data")

if os.path.exists(seg_dir):
    if selected_with_masks / df['Aneurysm Present'].sum() > 0.5:
        print(f"   ✓ Segmentation masks: GOOD coverage ({selected_with_masks / df['Aneurysm Present'].sum() * 100:.1f}%)")
    else:
        print(f"   ⚠ Segmentation masks: LOW coverage ({selected_with_masks / df['Aneurysm Present'].sum() * 100:.1f}%)")
        print(f"      Consider setting segmentation_weight=0.5 to reduce impact")
else:
    print(f"   ⚠ Segmentation masks: NOT AVAILABLE")
    print(f"      Training will use zero masks (uniform predictions expected)")
    print(f"      This is OK - detection and location are the priority tasks")

print("\n" + "=" * 80)
