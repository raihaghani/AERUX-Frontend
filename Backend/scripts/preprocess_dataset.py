import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Standalone Preprocessing Script for Intracranial Aneurysm Dataset

This script:
1. Loads DICOM files from the series directory
2. Filters out error series from error_data.yaml
3. Loads series from train.csv (full competition CSV)
4. Applies modality-specific preprocessing (N4, HU windowing, Sato vesselness, etc.)
5. Converts to 2.5D representation (7 slices)
6. Saves preprocessed volumes as .npy files

Directory structure:
    Input: D:/FYP_Data/rsna-intracranial-aneurysm-detection/series/{SeriesInstanceUID}/*.dcm
    Output: E:/Education/Aerux_Final/Preprocessed_images_2.5D/{modality}/{label}/{SeriesInstanceUID}.npy

Usage:
    python preprocess_dataset.py

"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import pydicom
import cv2
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config import Config

# Try to import preprocessing module
try:
    from src.datasets.preprocessing import create_preprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: Preprocessing module not available. Install: pip install SimpleITK scikit-image")
    PREPROCESSING_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_error_series_ids(yaml_path: str) -> set:
    """Load error series IDs from YAML file"""
    try:
        if not os.path.exists(yaml_path):
            logger.warning(f"Error data file not found: {yaml_path}")
            return set()
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            return set(data)
        else:
            logger.warning("Error data YAML format unexpected")
            return set()
    except Exception as e:
        logger.error(f"Error loading error_data.yaml: {e}")
        return set()


def load_localizer_data(localizers_csv_path: str) -> dict:
    """Load aneurysm slice locations from train_localizers.csv
    
    Returns:
        Dict mapping SeriesInstanceUID to dict of {SOPInstanceUID: (x, y) coordinates}
    """
    try:
        if not os.path.exists(localizers_csv_path):
            logger.warning(f"Localizers file not found: {localizers_csv_path}")
            return {}
        
        df = pd.read_csv(localizers_csv_path)
        
        # Create mapping: SeriesInstanceUID -> list of SOPInstanceUIDs with aneurysms
        localizer_dict = {}
        for _, row in df.iterrows():
            series_uid = row['SeriesInstanceUID']
            sop_uid = row['SOPInstanceUID']
            
            if series_uid not in localizer_dict:
                localizer_dict[series_uid] = []
            localizer_dict[series_uid].append(sop_uid)
        
        logger.info(f"Loaded localizer data for {len(localizer_dict)} series")
        return localizer_dict
        
    except Exception as e:
        logger.error(f"Error loading train_localizers.csv: {e}")
        return {}


def normalize_modality(modality: str) -> str:
    """Normalize modality names to CTA, MRA, or MRI"""
    modality_upper = str(modality).upper().strip()
    
    if 'CTA' in modality_upper:
        return 'CTA'
    elif 'MRA' in modality_upper:
        return 'MRA'
    elif 'T1' in modality_upper or 'T2' in modality_upper or 'MRI' in modality_upper:
        return 'MRI'
    else:
        return 'Unknown'


def load_dicom_series(series_dir: str) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[dict]]:
    """
    Load all DICOM files from a series directory and stack them into a 3D volume.
    
    Args:
        series_dir: Path to directory containing DICOM files
        
    Returns:
        Tuple of (3D volume, metadata dict, sop_to_index dict) or (None, None, None) if failed
        sop_to_index maps SOPInstanceUID to slice index in the volume
    """
    try:
        # Get all DICOM files
        dcm_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
        
        if not dcm_files:
            logger.warning(f"No DICOM files found in {series_dir}")
            return None, None, None
        
        # Load all slices
        slices = []
        metadata = None
        
        for dcm_file in dcm_files:
            dcm_path = os.path.join(series_dir, dcm_file)
            try:
                dcm = pydicom.dcmread(dcm_path)
                
                # Get pixel array
                img = dcm.pixel_array.astype(np.float32)
                
                # Apply rescale slope and intercept if available (for CT/CTA)
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
                
                # Store slice with position info and SOPInstanceUID
                if hasattr(dcm, 'ImagePositionPatient'):
                    position = float(dcm.ImagePositionPatient[2])
                else:
                    position = 0.0
                
                sop_uid = dcm.SOPInstanceUID if hasattr(dcm, 'SOPInstanceUID') else None
                slices.append((position, img, sop_uid))
                
                # Store metadata from first slice
                if metadata is None:
                    metadata = {
                        'Modality': dcm.Modality if hasattr(dcm, 'Modality') else 'Unknown',
                        'SliceThickness': float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0,
                        'PixelSpacing': [float(x) for x in dcm.PixelSpacing] if hasattr(dcm, 'PixelSpacing') else [1.0, 1.0],
                        'Rows': int(dcm.Rows) if hasattr(dcm, 'Rows') else img.shape[0],
                        'Columns': int(dcm.Columns) if hasattr(dcm, 'Columns') else img.shape[1]
                    }
            except Exception as e:
                logger.warning(f"Failed to load {dcm_file}: {e}")
                continue
        
        if not slices:
            return None, None, None
        
        # Sort slices by position
        slices.sort(key=lambda x: x[0])
        
        # Create SOPInstanceUID to index mapping
        sop_to_index = {}
        for idx, (_, _, sop_uid) in enumerate(slices):
            if sop_uid:
                sop_to_index[sop_uid] = idx
        
        # Stack into 3D volume
        volume = np.stack([s[1] for s in slices], axis=0)  # Shape: (num_slices, H, W)
        
        return volume, metadata, sop_to_index
        
    except Exception as e:
        logger.error(f"Error loading DICOM series from {series_dir}: {e}")
        return None, None, None


def extract_2_5d_slices(volume: np.ndarray, num_slices: int = 7, center_slice: Optional[int] = None) -> np.ndarray:
    """
    Extract 2.5D representation from 3D/4D volume.
    
    If center_slice is provided, extracts center_slice ± (num_slices-1)//2 slices.
    Otherwise, selects evenly spaced slices across the volume.
    
    Args:
        volume: 2D, 3D, or 4D volume
        num_slices: Number of slices to extract (default: 7)
        center_slice: Optional center slice index for aneurysm location
        
    Returns:
        2.5D volume (H, W, num_slices)
    """
    # Handle different volume shapes
    if volume.ndim == 2:
        # 2D image - replicate to create 2.5D
        height, width = volume.shape
        result = np.stack([volume] * num_slices, axis=-1)
        return result
    
    elif volume.ndim == 4:
        # 4D volume (batch, depth, height, width) - squeeze first dimension
        volume = np.squeeze(volume, axis=0)
        # Now it's 3D, continue processing
    
    if volume.ndim == 3:
        # Determine which dimension is depth
        shape = volume.shape
        
        # Find the dimension that's likely the depth
        # Usually depth is smaller than height/width
        min_dim_idx = np.argmin(shape)
        max_dim_idx = np.argmax(shape)
        
        # If smallest dimension is very small, it's likely the depth
        if shape[min_dim_idx] < shape[max_dim_idx] / 2:
            depth_idx = min_dim_idx
        else:
            # Otherwise assume first dimension is depth
            depth_idx = 0
        
        # Reorder to (depth, height, width)
        if depth_idx == 0:
            volume_ordered = volume
        elif depth_idx == 1:
            volume_ordered = np.transpose(volume, (1, 0, 2))
        else:  # depth_idx == 2
            volume_ordered = np.transpose(volume, (2, 0, 1))
        
        depth, height, width = volume_ordered.shape
        
        # Select slices based on center_slice or evenly spaced
        if center_slice is not None and depth >= num_slices:
            # Extract center_slice ± surrounding slices
            half_slices = (num_slices - 1) // 2
            
            # Ensure center_slice is within valid range
            center_slice = max(half_slices, min(center_slice, depth - half_slices - 1))
            
            # Extract slices around center
            start_idx = center_slice - half_slices
            end_idx = center_slice + half_slices + 1
            indices = list(range(start_idx, end_idx))
            
            # Ensure we have exactly num_slices
            if len(indices) < num_slices:
                # Pad with edge slices if needed
                while len(indices) < num_slices:
                    indices.append(indices[-1])
            elif len(indices) > num_slices:
                indices = indices[:num_slices]
                
        else:
            # Fallback to evenly spaced slices
            if depth < num_slices:
                # If fewer slices than needed, repeat slices
                indices = np.linspace(0, depth - 1, num_slices).astype(int)
            else:
                indices = np.linspace(0, depth - 1, num_slices).astype(int)
        
        # Extract slices
        selected_slices = volume_ordered[indices]  # Shape: (num_slices, H, W)
        
        # Transpose to (H, W, num_slices)
        result = np.transpose(selected_slices, (1, 2, 0))
        
        return result
    
    else:
        raise ValueError(f"Unsupported volume dimensions: {volume.shape}")


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize 2.5D volume to target spatial dimensions.
    
    Args:
        volume: Input volume (H, W, C)
        target_size: Target (height, width)
        
    Returns:
        Resized volume (target_H, target_W, C)
    """
    if volume.shape[:2] == target_size:
        return volume
    
    # Resize each channel separately
    num_channels = volume.shape[2] if volume.ndim == 3 else 1
    
    if volume.ndim == 2:
        resized = cv2.resize(volume, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_channels = []
        for i in range(num_channels):
            channel = cv2.resize(volume[:, :, i], target_size, interpolation=cv2.INTER_LINEAR)
            resized_channels.append(channel)
        resized = np.stack(resized_channels, axis=-1)
    
    return resized


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range"""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 1e-6:
        return (volume - vmin) / (vmax - vmin)
    return np.zeros_like(volume)


def preprocess_series(
    series_uid: str,
    series_dir: str,
    modality: str,
    output_path: str,
    config: Config,
    preprocessor=None,
    target_size: Tuple[int, int] = (256, 256),
    num_slices: int = 7,
    aneurysm_sop_uids: Optional[List[str]] = None
) -> bool:
    """
    Preprocess a single series.
    
    Args:
        series_uid: SeriesInstanceUID
        series_dir: Path to series directory with DICOM files
        modality: Modality (CTA, MRA, MRI)
        output_path: Path to save preprocessed .npy file
        config: Configuration object
        preprocessor: Preprocessing object (optional)
        target_size: Target image size
        num_slices: Number of slices for 2.5D
        aneurysm_sop_uids: Optional list of SOPInstanceUIDs containing aneurysms
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load DICOM series with SOPInstanceUID mapping
        volume_3d, metadata, sop_to_index = load_dicom_series(series_dir)
        
        if volume_3d is None:
            logger.warning(f"Failed to load series: {series_uid}")
            return False
        
        # Check volume shape
        if volume_3d.size == 0:
            logger.warning(f"Empty volume for series: {series_uid}")
            return False
        
        # Find center slice from aneurysm SOPInstanceUIDs
        center_slice = None
        if aneurysm_sop_uids and sop_to_index:
            # Find indices of slices with aneurysms
            aneurysm_indices = []
            for sop_uid in aneurysm_sop_uids:
                if sop_uid in sop_to_index:
                    aneurysm_indices.append(sop_to_index[sop_uid])
            
            # Use median slice if multiple aneurysms
            if aneurysm_indices:
                center_slice = int(np.median(aneurysm_indices))
        
        # Extract 2.5D slices (using center_slice if found)
        try:
            volume_2_5d = extract_2_5d_slices(volume_3d, num_slices=num_slices, center_slice=center_slice)
        except Exception as e:
            logger.error(f"Failed to extract 2.5D slices for {series_uid} (shape: {volume_3d.shape}): {e}")
            return False
        
        # Resize to target size
        volume_2_5d = resize_volume(volume_2_5d, target_size=target_size)
        
        # Apply preprocessing if available
        if preprocessor is not None and PREPROCESSING_AVAILABLE:
            try:
                # Process each slice
                preprocessed_slices = []
                for i in range(volume_2_5d.shape[2]):
                    slice_2d = volume_2_5d[:, :, i]
                    result = preprocessor.preprocess(slice_2d)
                    preprocessed_slices.append(result['preprocessed'])
                
                volume_2_5d = np.stack(preprocessed_slices, axis=-1)
            except Exception as e:
                logger.warning(f"Preprocessing failed for {series_uid}: {e}. Using original.")
        
        # Normalize to [0, 1]
        volume_2_5d = normalize_volume(volume_2_5d)
        
        # Save as .npy
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, volume_2_5d.astype(np.float32))
        
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing {series_uid}: {e}")
        return False


def main():
    """Main preprocessing pipeline"""
    
    print("=" * 80)
    print("INTRACRANIAL ANEURYSM DATASET PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load configuration
    config = Config()
    
    # Display configuration
    print(f"\nInput Directory: {config.data.series_dir}")
    print(f"Output Directories:")
    print(f"  CTA: {config.data.preprocessed_cta_dir}")
    print(f"  MRA: {config.data.preprocessed_mra_dir}")
    print(f"  MRI: {config.data.preprocessed_mri_dir}")
    print(f"\nTarget Size: {config.data.target_size}")
    print(f"Number of Slices: {config.data.num_slices}")
    print(f"Apply Preprocessing: {config.data.apply_preprocessing}")
    
    # Load localizer data for aneurysm slice locations
    print(f"\nLoading aneurysm localizer data...")
    localizer_dict = load_localizer_data(config.data.train_localizers_csv)
    print(f"Loaded slice locations for {len(localizer_dict)} series with aneurysms")
    
    print(f"\nLoading full dataset from: {config.data.train_csv}")
    df = pd.read_csv(config.data.train_csv)
    print(f"Total series count: {len(df)}")

    # Filter error series
    if os.path.exists(config.data.error_data_yaml):
        error_series = load_error_series_ids(config.data.error_data_yaml)
        if error_series:
            original_count = len(df)
            df = df[~df['SeriesInstanceUID'].isin(error_series)]
            print(f"Filtered out {original_count - len(df)} error series")
            print(f"Remaining series: {len(df)}")
    
    # Normalize modality names
    df['Modality_Normalized'] = df['Modality'].apply(normalize_modality)
    
    # Remove unknown modalities
    df = df[df['Modality_Normalized'] != 'Unknown']
    
    # Display distribution
    print(f"\nModality Distribution:")
    for modality, count in df['Modality_Normalized'].value_counts().items():
        print(f"  {modality}: {count}")
    
    print(f"\nLabel Distribution:")
    for label, count in df['Aneurysm Present'].value_counts().items():
        label_name = "Aneurysm" if label == 1 else "No Aneurysm"
        print(f"  {label_name}: {count}")
    
    # Initialize preprocessors
    preprocessors = {}
    if config.data.apply_preprocessing and PREPROCESSING_AVAILABLE:
        print("\nInitializing preprocessors...")
        try:
            # MRI/MRA preprocessor
            mri_config = {
                'apply_n4': config.data.mri_apply_n4,
                'apply_vesselness': config.data.mri_apply_vesselness,
                'apply_otsu': config.data.mri_apply_otsu,
                'vesselness_scale_range': config.data.mri_vesselness_scale_range,
                'vesselness_scale_step': config.data.mri_vesselness_scale_step,
                'n4_max_iterations': config.data.mri_n4_max_iterations
            }
            
            # CTA preprocessor
            cta_config = {
                'apply_hu_windowing': config.data.cta_apply_hu_windowing,
                'apply_clahe': config.data.cta_apply_clahe,
                'apply_vesselness': config.data.cta_apply_vesselness,
                'window_center': config.data.cta_window_center,
                'window_width': config.data.cta_window_width,
                'clahe_clip_limit': config.data.cta_clahe_clip_limit,
                'clahe_tile_grid_size': config.data.cta_clahe_tile_grid_size,
                'vesselness_scale_range': config.data.cta_vesselness_scale_range,
                'vesselness_scale_step': config.data.cta_vesselness_scale_step
            }
            
            preprocessors['MRI'] = create_preprocessor('MRI', **mri_config)
            preprocessors['MRA'] = create_preprocessor('MRA', **mri_config)
            preprocessors['CTA'] = create_preprocessor('CTA', **cta_config)
            
            print(" Preprocessors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize preprocessors: {e}")
            print(" Preprocessing will be skipped")
    else:
        print("\nPreprocessing disabled or not available")
    
    # Create output directories
    print("\nCreating output directories...")
    for modality in ['CTA', 'MRA', 'MRI']:
        for label in ['aneurysm', 'no_aneurysm']:
            if modality == 'CTA':
                base_dir = config.data.preprocessed_cta_dir
            elif modality == 'MRA':
                base_dir = config.data.preprocessed_mra_dir
            else:
                base_dir = config.data.preprocessed_mri_dir
            
            output_dir = os.path.join(base_dir, label)
            os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total': len(df),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'by_modality': {'CTA': 0, 'MRA': 0, 'MRI': 0}
    }
    
    # Process each series
    print("\n" + "=" * 80)
    print("STARTING PREPROCESSING")
    print("=" * 80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing series"):
        series_uid = row['SeriesInstanceUID']
        modality = row['Modality_Normalized']
        label = int(row['Aneurysm Present'])
        label_name = 'aneurysm' if label == 1 else 'no_aneurysm'
        
        # Construct paths
        series_dir = os.path.join(config.data.series_dir, series_uid)
        
        if modality == 'CTA':
            base_dir = config.data.preprocessed_cta_dir
        elif modality == 'MRA':
            base_dir = config.data.preprocessed_mra_dir
        else:
            base_dir = config.data.preprocessed_mri_dir
        
        output_path = os.path.join(base_dir, label_name, f"{series_uid}.npy")
        
        # Skip if already preprocessed
        if os.path.exists(output_path):
            logger.info(f"Skipping already preprocessed series: {series_uid}")
            stats['skipped'] += 1
            continue
        
        # Check if series directory exists
        if not os.path.exists(series_dir):
            logger.warning(f"Series directory not found: {series_dir}")
            stats['failed'] += 1
            continue
        
        # Get preprocessor
        preprocessor = preprocessors.get(modality, None)
        
        # Get aneurysm SOPInstanceUIDs from localizer data if available
        aneurysm_sop_uids = localizer_dict.get(series_uid, None)
        
        # Preprocess
        success = preprocess_series(
            series_uid=series_uid,
            series_dir=series_dir,
            modality=modality,
            output_path=output_path,
            config=config,
            preprocessor=preprocessor,
            target_size=config.data.target_size,
            num_slices=config.data.num_slices,
            aneurysm_sop_uids=aneurysm_sop_uids
        )
        
        if success:
            stats['successful'] += 1
            stats['by_modality'][modality] += 1
        else:
            stats['failed'] += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)
    print(f"\nTotal series: {stats['total']}")
    print(f"Successfully preprocessed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    
    print(f"\nBy Modality:")
    for modality, count in stats['by_modality'].items():
        print(f"  {modality}: {count}")
    
    print(f"\nOutput directories:")
    print(f"  CTA: {config.data.preprocessed_cta_dir}")
    print(f"  MRA: {config.data.preprocessed_mra_dir}")
    print(f"  MRI: {config.data.preprocessed_mri_dir}")
    
    print("\n" + "=" * 80)
    print("Log file saved to: preprocessing_pipeline.log")
    print("=" * 80)


if __name__ == "__main__":
    main()
