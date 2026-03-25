"""
Preprocessing Comparison Visualization

Visualizes original DICOM images vs preprocessed 2.5D volumes for CTA, MRA, and MRI.
Shows the effect of preprocessing (HU windowing for CTA, vesselness for MRA/MRI).

Usage:
    # Visualize specific series
    python visualize_preprocessing.py --series_uid 1.2.826.0.1.3680043...
    
    # Visualize random samples from each modality
    python visualize_preprocessing.py --random --n_samples 5
    
    # Save all visualizations
    python visualize_preprocessing.py --save_all --output_dir preprocessing_viz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.config import Config


def load_original_dicom(series_uid, series_dir, aneurysm_sop_uid=None):
    """Load original DICOM series and return aneurysm slice or center slice"""
    series_path = os.path.join(series_dir, series_uid)
    
    if not os.path.exists(series_path):
        print(f"Series directory not found: {series_path}")
        return None, None, None
    
    # Load all DICOM files
    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    if len(dicom_files) == 0:
        print(f"No DICOM files found in {series_path}")
        return None, None, None
    
    # Load and sort by slice location
    slices = []
    sop_to_idx = {}
    for fname in dicom_files:
        try:
            ds = pydicom.dcmread(os.path.join(series_path, fname))
            slices.append(ds)
            if hasattr(ds, 'SOPInstanceUID'):
                sop_to_idx[ds.SOPInstanceUID] = len(slices) - 1
        except:
            continue
    
    # Sort by SliceLocation or InstanceNumber
    if hasattr(slices[0], 'SliceLocation'):
        slices.sort(key=lambda x: float(x.SliceLocation))
    elif hasattr(slices[0], 'InstanceNumber'):
        slices.sort(key=lambda x: int(x.InstanceNumber))
    
    # Rebuild SOP to index mapping after sorting
    sop_to_idx = {}
    for idx, ds in enumerate(slices):
        if hasattr(ds, 'SOPInstanceUID'):
            sop_to_idx[ds.SOPInstanceUID] = idx
    
    # Find aneurysm slice if SOP UID provided
    slice_idx = None
    if aneurysm_sop_uid and aneurysm_sop_uid in sop_to_idx:
        slice_idx = sop_to_idx[aneurysm_sop_uid]
        print(f"  Found aneurysm at slice {slice_idx}/{len(slices)}")
    else:
        # Default to center slice
        slice_idx = len(slices) // 2
        print(f"  Using center slice {slice_idx}/{len(slices)} (aneurysm location unknown)")
    
    target_slice = slices[slice_idx].pixel_array.astype(np.float32)
    
    # Get modality
    modality = slices[0].Modality if hasattr(slices[0], 'Modality') else "Unknown"
    
    return target_slice, modality, slice_idx


def load_preprocessed_volume(series_uid, modality, base_dirs):
    """Load preprocessed 2.5D volume and return center slice (where aneurysm is)"""
    for label_dir in ['aneurysm', 'no_aneurysm']:
        for mod_name in ['CTA', 'MRA', 'MRI']:
            if modality.upper().startswith(mod_name) or mod_name in modality.upper():
                base_dir = base_dirs.get(mod_name)
                if base_dir is None:
                    continue
                    
                file_path = os.path.join(base_dir, label_dir, f"{series_uid}.npy")
                if os.path.exists(file_path):
                    volume = np.load(file_path)  # Shape: (7, 256, 256)
                    # Center slice (4th of 7) is where aneurysm is located during preprocessing
                    center_slice = volume[3]
                    return center_slice, mod_name
    
    return None, None


def get_aneurysm_sop_uid(series_uid, localizers_csv):
    """Get SOPInstanceUID where aneurysm is located"""
    try:
        df = pd.read_csv(localizers_csv)
        # Filter by series
        series_data = df[df['SeriesInstanceUID'] == series_uid]
        if len(series_data) > 0:
            # Get first SOPInstanceUID (where aneurysm is)
            sop_uid = series_data.iloc[0]['SOPInstanceUID']
            return sop_uid
    except Exception as e:
        print(f"Could not load localizers: {e}")
    return None


def visualize_comparison(series_uid, original_slice, preprocessed_slice, modality, slice_idx=None, save_path=None):
    """Create side-by-side comparison visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    slice_info = f" (Slice {slice_idx})" if slice_idx is not None else ""
    fig.suptitle(
        f"Preprocessing Comparison - {modality}{slice_info}\n"
        f"Series: {series_uid[:50]}...",
        fontsize=13, fontweight='bold'
    )
    
    # Original slice
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f'Original {modality} (DICOM)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Add preprocessing info
    if modality == 'CT' or 'CTA' in modality:
        preprocessing_info = "Preprocessing:\n• N4 Bias Correction\n• HU Windowing (40±60)\n• Normalization [0,1]\n• Resize to 256×256"
    else:  # MRI/MRA
        preprocessing_info = "Preprocessing:\n• N4 Bias Correction\n• Vesselness Filter\n• Normalization [0,1]\n• Resize to 256×256"
    
    axes[0].text(
        0.02, 0.98, preprocessing_info,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Preprocessed slice
    axes[1].imshow(preprocessed_slice, cmap='gray')
    axes[1].set_title('Preprocessed 2.5D (Center Slice)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add statistics
    stats_text = (
        f"Shape: {preprocessed_slice.shape}\n"
        f"Min: {preprocessed_slice.min():.4f}\n"
        f"Max: {preprocessed_slice.max():.4f}\n"
        f"Mean: {preprocessed_slice.mean():.4f}\n"
        f"Std: {preprocessed_slice.std():.4f}"
    )
    axes[1].text(
        0.98, 0.98, stats_text,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def visualize_all_modalities(series_uids, config, save_dir=None):
    """Visualize CTA, MRA, and MRI side by side"""
    
    base_dirs = {
        'CTA': config.data.preprocessed_cta_dir,
        'MRA': config.data.preprocessed_mra_dir,
        'MRI': config.data.preprocessed_mri_dir
    }
    
    series_dir = config.data.series_dir
    localizers_csv = config.data.train_localizers_csv
    
    # Try to find one example of each modality (prefer aneurysm cases)
    examples = {'CTA': None, 'MRA': None, 'MRI': None}
    
    for uid in series_uids:
        for mod in ['CTA', 'MRA', 'MRI']:
            if examples[mod] is not None:
                continue
                
            # Try to load preprocessed
            preprocessed, detected_mod = load_preprocessed_volume(uid, mod, base_dirs)
            if preprocessed is not None and detected_mod == mod:
                # Get aneurysm location
                sop_uid = get_aneurysm_sop_uid(uid, localizers_csv)
                
                # Try to load original at aneurysm slice
                original, dicom_mod, slice_idx = load_original_dicom(uid, series_dir, sop_uid)
                if original is not None:
                    examples[mod] = {
                        'uid': uid,
                        'original': original,
                        'preprocessed': preprocessed,
                        'modality': detected_mod,
                        'slice_idx': slice_idx,
                        'has_aneurysm': sop_uid is not None
                    }
                    aneurysm_info = "with aneurysm" if sop_uid else "center slice"
                    print(f"Found {mod} example ({aneurysm_info}): {uid[:40]}...")
                    break
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Preprocessing Comparison: All Modalities', fontsize=16, fontweight='bold')
    
    for idx, (mod, data) in enumerate(examples.items()):
        if data is None:
            # No example found
            axes[0, idx].text(0.5, 0.5, f'No {mod}\nExample Found', 
                            ha='center', va='center', fontsize=14,
                            transform=axes[0, idx].transAxes)
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
            continue
        
        # Original
        axes[0, idx].imshow(data['original'], cmap='gray')
        slice_info = f" (Slice {data['slice_idx']})" if data.get('slice_idx') is not None else ""
        aneurysm_marker = " 🔴" if data.get('has_aneurysm') else ""
        axes[0, idx].set_title(f'{mod} - Original DICOM{slice_info}{aneurysm_marker}', 
                              fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Preprocessed
        axes[1, idx].imshow(data['preprocessed'], cmap='gray')
        axes[1, idx].set_title(f'{mod} - Preprocessed (Center Slice){aneurysm_marker}', 
                              fontsize=11, fontweight='bold')
        axes[1, idx].axis('off')
        
        # Add preprocessing info
        if mod == 'CTA':
            info = "• HU Windowing\n• N4 Correction"
        else:
            info = "• Vesselness Filter\n• N4 Correction"
        
        if data.get('has_aneurysm'):
            info += "\n🔴 Aneurysm Present"
        
        axes[1, idx].text(
            0.02, 0.98, info,
            transform=axes[1, idx].transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'all_modalities_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Preprocessing Effects')
    parser.add_argument('--series_uid', type=str, default=None,
                       help='Specific series UID to visualize')
    parser.add_argument('--random', action='store_true',
                       help='Visualize random samples')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of random samples per modality (default: 5)')
    parser.add_argument('--all_modalities', action='store_true',
                       help='Show CTA, MRA, MRI side-by-side comparison')
    parser.add_argument('--save_all', action='store_true',
                       help='Save all visualizations')
    parser.add_argument('--output_dir', type=str, default='preprocessing_viz',
                       help='Output directory for saved visualizations')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    base_dirs = {
        'CTA': config.data.preprocessed_cta_dir,
        'MRA': config.data.preprocessed_mra_dir,
        'MRI': config.data.preprocessed_mri_dir
    }
    
    series_dir = config.data.series_dir
    
    # Create output directory
    if args.save_all:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of available series
    try:
        train_df = pd.read_csv(config.data.train_csv)
        available_series = train_df['SeriesInstanceUID'].unique().tolist()
    except:
        # Fallback: scan preprocessed directories
        available_series = []
        for mod_dir in base_dirs.values():
            for label in ['aneurysm', 'no_aneurysm']:
                label_dir = os.path.join(mod_dir, label)
                if os.path.exists(label_dir):
                    files = [f.replace('.npy', '') for f in os.listdir(label_dir) if f.endswith('.npy')]
                    available_series.extend(files)
        available_series = list(set(available_series))
    
    print(f"Found {len(available_series)} available series")
    
    # All modalities comparison
    if args.all_modalities:
        print("\nCreating all-modalities comparison...")
        visualize_all_modalities(
            available_series[:100],  # Check first 100
            config,
            args.output_dir if args.save_all else None
        )
        return
    
    # Specific series
    if args.series_uid:
        print(f"\nVisualizing series: {args.series_uid}")
        
        # Get aneurysm location if available
        localizers_csv = config.data.train_localizers_csv
        sop_uid = get_aneurysm_sop_uid(args.series_uid, localizers_csv)
        
        if sop_uid:
            print(f"Found aneurysm location: {sop_uid[:40]}...")
        
        # Load original at aneurysm slice
        original, modality, slice_idx = load_original_dicom(args.series_uid, series_dir, sop_uid)
        if original is None:
            print("Failed to load original DICOM")
            return
        
        # Load preprocessed
        preprocessed, detected_mod = load_preprocessed_volume(args.series_uid, modality, base_dirs)
        if preprocessed is None:
            print("Failed to load preprocessed volume")
            return
        
        save_path = os.path.join(args.output_dir, f"{args.series_uid[:30]}.png") if args.save_all else None
        visualize_comparison(args.series_uid, original, preprocessed, detected_mod, slice_idx, save_path)
    
    # Random samples
    elif args.random:
        import random
        
        # Sample per modality
        for modality in ['CTA', 'MRA', 'MRI']:
            print(f"\n=== {modality} Samples ===")
            
            # Find series with this modality
            mod_series = []
            base_dir = base_dirs[modality]
            for label in ['aneurysm', 'no_aneurysm']:
                label_dir = os.path.join(base_dir, label)
                if os.path.exists(label_dir):
                    files = [f.replace('.npy', '') for f in os.listdir(label_dir) if f.endswith('.npy')]
                    mod_series.extend(files)
            
            if len(mod_series) == 0:
                print(f"No {modality} samples found")
                continue
            
            # Random sample
            n = min(args.n_samples, len(mod_series))
            sampled = random.sample(mod_series, n)
            
            for i, uid in enumerate(sampled):
                print(f"[{i+1}/{n}] Processing {uid[:40]}...")
                
                # Get aneurysm location
                localizers_csv = config.data.train_localizers_csv
                sop_uid = get_aneurysm_sop_uid(uid, localizers_csv)
                
                # Load original at aneurysm slice
                original, _, slice_idx = load_original_dicom(uid, series_dir, sop_uid)
                preprocessed, _ = load_preprocessed_volume(uid, modality, base_dirs)
                
                if original is None or preprocessed is None:
                    print(f"  Skipping (failed to load)")
                    continue
                
                save_path = os.path.join(args.output_dir, f"{modality}_{i+1:02d}_{uid[:20]}.png") if args.save_all else None
                visualize_comparison(uid, original, preprocessed, modality, slice_idx, save_path)
    
    else:
        print("Please specify --series_uid, --random, or --all_modalities")
        print("\nExample usage:")
        print("  python visualize_preprocessing.py --all_modalities --save_all")
        print("  python visualize_preprocessing.py --random --n_samples 3 --save_all")
    
    if args.save_all:
        print("\n" + "=" * 80)
        print(f"[OK] All visualizations saved to: {args.output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
