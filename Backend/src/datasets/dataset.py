"""
Data Loading and Preprocessing Module
Handles multi-modal medical imaging data (CTA, MRA, MRI) with support for
detection, localization (12 aneurysm locations), and segmentation tasks.

Note: MRI T1post and MRI T2 are treated as a single 'MRI' modality.
Series lists come from train.csv (or filtered splits such as global_train_series.csv).

Author: Senior Computer Vision Engineer
Date: 2025-01-08
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import json
import ast

# Import preprocessing module
try:
    from src.datasets.preprocessing import create_preprocessor
except ImportError:
    print("Warning: Preprocessing module not available. Install required packages: pip install SimpleITK scikit-image")
    create_preprocessor = None


class AneurysmDataset(Dataset):
    """
    Multi-modal dataset for intracranial aneurysm detection, localization, and segmentation.
    
    Supports:
    - Multi-modal imaging: CTA, MRA, MRI (T1post and T2 combined)
    - Binary detection: Aneurysm present or not
    - 12-class localization: Specific aneurysm location
    - Segmentation: Pixel-level aneurysm mask
    
    """
    
    def __init__(
        self,
        data_info: pd.DataFrame,
        modality_dirs: Dict[str, str],
        location_labels: List[str],
        segmentation_dir: Optional[str] = None,
        target_size: Tuple[int, int] = (256, 256),
        num_slices: int = 7,
        transform=None,
        is_training: bool = True,
        apply_preprocessing: bool = False,
        preprocessing_config: Optional[Dict] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_info: DataFrame with columns [SeriesInstanceUID, modality, detection_label, location_labels]
            modality_dirs: Dictionary mapping modality names to preprocessed data directories
            location_labels: List of 12 aneurysm location names
            segmentation_dir: Directory containing segmentation masks (optional)
            target_size: Target image size (H, W)
            num_slices: Number of slices for 2.5D representation
            transform: Data augmentation transforms
            is_training: Whether this is training data (for augmentation)
            apply_preprocessing: Whether to apply preprocessing techniques (N4, HU windowing, vesselness, etc.)
            preprocessing_config: Dictionary with preprocessing parameters for each modality
        """
        self.data_info = data_info
        self.modality_dirs = modality_dirs
        self.location_labels = location_labels
        self.segmentation_dir = segmentation_dir
        self.target_size = target_size
        self.num_slices = num_slices
        self.transform = transform
        self.is_training = is_training
        self.apply_preprocessing = apply_preprocessing
        self.preprocessing_config = preprocessing_config or {}
        
        # Create location label to index mapping
        self.location_to_idx = {label: idx for idx, label in enumerate(location_labels)}
        self.location_to_idx["No Aneurysm"] = len(location_labels)  # Index 12 for no aneurysm
        
        # Initialize preprocessors if enabled
        self.preprocessors = {}
        if self.apply_preprocessing and create_preprocessor is not None:
            self._initialize_preprocessors()
        
    def __len__(self) -> int:
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
            - 'cta': CTA image tensor (C, H, W) or None
            - 'mra': MRA image tensor (C, H, W) or None
            - 'mri': MRI image tensor (C, H, W) or None (T1post or T2)
            - 'detection_label': Binary label (0=no aneurysm, 1=aneurysm)
            - 'location_label': Location class index (0-12)
            - 'location_multi_hot': Multi-hot encoding for multiple locations
            - 'segmentation_mask': Segmentation mask (H, W) or None
            - 'series_uid': SeriesInstanceUID
            - 'modality': Primary modality
        """
        row = self.data_info.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        modality = row['Modality']
        detection_label = int(row['Aneurysm Present'])
        
        # Initialize sample dictionary
        sample = {
            'cta': None,
            'mra': None,
            'mri': None,  # Simplified: T1 and T2 both map to 'mri'
            'detection_label': detection_label,
            'series_uid': series_uid,
            'modality': modality
        }
        
        # Normalize modality name (treat T1post and T2 as MRI)
        modality_normalized = self._normalize_modality(modality)
        
        # Load primary modality image
        if modality_normalized == 'CTA' and 'CTA' in self.modality_dirs:
            sample['cta'] = self._load_image(series_uid, self.modality_dirs['CTA'], detection_label, 'CTA')
        elif modality_normalized == 'MRA' and 'MRA' in self.modality_dirs:
            sample['mra'] = self._load_image(series_uid, self.modality_dirs['MRA'], detection_label, 'MRA')
        elif modality_normalized == 'MRI' and 'MRI' in self.modality_dirs:
            sample['mri'] = self._load_image(series_uid, self.modality_dirs['MRI'], detection_label, 'MRI')
        
        # Load all available modalities (if multi-modal training)
        for mod_key, mod_dir in self.modality_dirs.items():
            mod_key_lower = mod_key.lower()
            if sample.get(mod_key_lower) is None:
                # Determine modality name for preprocessing
                mod_name = mod_key if mod_key in ['CTA', 'MRA', 'MRI'] else 'MRI'
                
                img = self._load_image(series_uid, mod_dir, detection_label, mod_name)
                if img is not None:
                    sample[mod_key_lower] = img
        
        # Process location labels
        location_label, location_multi_hot = self._process_location_labels(row)
        sample['location_label'] = location_label
        sample['location_multi_hot'] = location_multi_hot
        
        # Load segmentation mask if available
        if self.segmentation_dir and detection_label == 1:
            sample['segmentation_mask'] = self._load_segmentation_mask(series_uid)
        else:
            # Create empty mask for no aneurysm cases
            sample['segmentation_mask'] = torch.zeros(self.target_size, dtype=torch.float32)
        
        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_image(
        self, 
        series_uid: str, 
        modality_dir: str, 
        detection_label: int,
        modality: str = None
    ) -> Optional[torch.Tensor]:
        """
        Load preprocessed 2.5D image with optional on-the-fly preprocessing
        
        Args:
            series_uid: SeriesInstanceUID
            modality_dir: Directory containing preprocessed images
            detection_label: 0 or 1 (no_aneurysm or aneurysm)
            modality: Imaging modality for preprocessing (CTA, MRA, MRI, etc.)
            
        Returns:
            Image tensor of shape (num_slices, H, W) or None if not found
        """
        # Construct file path
        label_folder = 'aneurysm' if detection_label == 1 else 'no_aneurysm'
        file_path = os.path.join(modality_dir, label_folder, f"{series_uid}.npy")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # Load numpy array (expected shape: H, W, num_slices)
            img_data = np.load(file_path)
            
            # Ensure correct dimensions
            if img_data.ndim == 2:
                # Single slice, replicate to create 2.5D
                img_data = np.stack([img_data] * self.num_slices, axis=-1)
            elif img_data.ndim == 3:
                # Already 2.5D, ensure correct number of slices
                if img_data.shape[-1] != self.num_slices:
                    # Resize along slice dimension
                    img_data = self._resize_slices(img_data, self.num_slices)
            
            # Ensure correct spatial size
            if img_data.shape[:2] != self.target_size:
                img_data = cv2.resize(
                    img_data.reshape(img_data.shape[0], img_data.shape[1], -1),
                    self.target_size,
                    interpolation=cv2.INTER_LINEAR
                ).reshape(*self.target_size, -1)
            
            # Apply modality-specific preprocessing if enabled
            if modality and self.apply_preprocessing:
                img_data = self._apply_preprocessing(img_data, modality)
            
            # Normalize to [0, 1]
            img_data = self._normalize(img_data)
            
            # Convert to tensor: (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_data).float().permute(2, 0, 1)
            
            return img_tensor
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None
    
    def _load_segmentation_mask(self, series_uid: str) -> torch.Tensor:
        """
        Load segmentation mask if available
        
        Args:
            series_uid: SeriesInstanceUID
            
        Returns:
            Binary mask tensor of shape (H, W)
        """
        if self.segmentation_dir is None:
            return torch.zeros(self.target_size, dtype=torch.float32)
        
        # Try to find mask file
        mask_path = os.path.join(self.segmentation_dir, f"{series_uid}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.segmentation_dir, f"{series_uid}.npy")
        
        if os.path.exists(mask_path):
            try:
                if mask_path.endswith('.png'):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.load(mask_path)
                
                # Resize to target size
                if mask.shape != self.target_size:
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Binarize
                mask = (mask > 0).astype(np.float32)
                
                return torch.from_numpy(mask).float()
                
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
        
        # Return empty mask if not found
        return torch.zeros(self.target_size, dtype=torch.float32)
    
    def _process_location_labels(self, row: pd.Series) -> Tuple[int, torch.Tensor]:
        """
        Process location labels from CSV row
        
        Args:
            row: DataFrame row containing location columns
            
        Returns:
            - Primary location index (0-12)
            - Multi-hot encoding tensor (13,) for all locations
        """
        # Create multi-hot encoding
        multi_hot = torch.zeros(len(self.location_labels) + 1, dtype=torch.float32)
        
        # Check each location column
        primary_location = len(self.location_labels)  # Default: No Aneurysm
        
        for i, location in enumerate(self.location_labels):
            if location in row and int(row[location]) == 1:
                multi_hot[i] = 1.0
                if primary_location == len(self.location_labels):  # Set first detected location as primary
                    primary_location = i
        
        # If no location detected, set "No Aneurysm"
        if multi_hot.sum() == 0:
            multi_hot[-1] = 1.0
        
        return primary_location, multi_hot
    
    def _resize_slices(self, img: np.ndarray, target_slices: int) -> np.ndarray:
        """
        Resize image along slice dimension
        
        Args:
            img: Image array of shape (H, W, num_slices)
            target_slices: Target number of slices
            
        Returns:
            Resized image of shape (H, W, target_slices)
        """
        H, W, current_slices = img.shape
        
        if current_slices == target_slices:
            return img
        
        # Use interpolation along slice dimension
        indices = np.linspace(0, current_slices - 1, target_slices)
        
        # Interpolate each spatial location
        resized = np.zeros((H, W, target_slices), dtype=img.dtype)
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), current_slices - 1)
            weight = idx - idx_low
            
            if idx_low == idx_high:
                resized[:, :, i] = img[:, :, idx_low]
            else:
                resized[:, :, i] = (1 - weight) * img[:, :, idx_low] + weight * img[:, :, idx_high]
        
        return resized
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            img: Input image array
            
        Returns:
            Normalized image
        """
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min < 1e-6:
            return np.zeros_like(img)
        
        return (img - img_min) / (img_max - img_min)
    
    @staticmethod
    def _normalize_modality(modality: str) -> str:
        """
        Normalize modality names. Treats MRI T1post and MRI T2 as a single 'MRI' category.
        
        Args:
            modality: Original modality string
            
        Returns:
            Normalized modality name ('CTA', 'MRA', or 'MRI')
        """
        modality_upper = str(modality).upper().strip()
        
        if 'CTA' in modality_upper:
            return 'CTA'
        elif 'MRA' in modality_upper:
            return 'MRA'
        elif 'T1' in modality_upper or 'T2' in modality_upper or 'MRI' in modality_upper:
            return 'MRI'
        else:
            return 'MRI'  # Default to MRI for unknown
    
    def _initialize_preprocessors(self):
        """
        Initialize preprocessing pipelines for each modality based on configuration.
        
        Creates MRI/MRA and CTA preprocessors with parameters from preprocessing_config.
        """
        try:
            # MRI/MRA preprocessors
            mri_config = {
                'apply_n4': self.preprocessing_config.get('mri_apply_n4', True),
                'apply_vesselness': self.preprocessing_config.get('mri_apply_vesselness', True),
                'apply_otsu': self.preprocessing_config.get('mri_apply_otsu', False),
                'vesselness_scale_range': tuple(self.preprocessing_config.get('mri_vesselness_scale_range', (1, 8))),
                'vesselness_scale_step': self.preprocessing_config.get('mri_vesselness_scale_step', 2),
                'n4_max_iterations': self.preprocessing_config.get('mri_n4_max_iterations', [50, 50, 50, 50])
            }
            
            # CTA preprocessor
            cta_config = {
                'apply_hu_windowing': self.preprocessing_config.get('cta_apply_hu_windowing', True),
                'apply_clahe': self.preprocessing_config.get('cta_apply_clahe', False),
                'apply_vesselness': self.preprocessing_config.get('cta_apply_vesselness', True),
                'window_center': self.preprocessing_config.get('cta_window_center', 40.0),
                'window_width': self.preprocessing_config.get('cta_window_width', 80.0),
                'clahe_clip_limit': self.preprocessing_config.get('cta_clahe_clip_limit', 3.0),
                'clahe_tile_grid_size': tuple(self.preprocessing_config.get('cta_clahe_tile_grid_size', (8, 8))),
                'vesselness_scale_range': tuple(self.preprocessing_config.get('cta_vesselness_scale_range', (1, 8))),
                'vesselness_scale_step': self.preprocessing_config.get('cta_vesselness_scale_step', 2)
            }
            
            # Create preprocessors (simplified: single MRI preprocessor for all MRI types)
            self.preprocessors['MRI'] = create_preprocessor('MRI', **mri_config)
            self.preprocessors['MRA'] = create_preprocessor('MRA', **mri_config)
            self.preprocessors['CTA'] = create_preprocessor('CTA', **cta_config)
            
            print("✓ Preprocessing pipelines initialized successfully")
            
        except Exception as e:
            print(f"Warning: Failed to initialize preprocessors: {e}")
            self.preprocessors = {}
    
    def _apply_preprocessing(self, img: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply preprocessing techniques based on modality.
        
        Args:
            img: Input image array (H, W) or (H, W, C)
            modality: Imaging modality (CTA, MRA, MRI, etc.)
            
        Returns:
            Preprocessed image with same shape as input
        """
        if not self.apply_preprocessing or modality not in self.preprocessors:
            return img
        
        try:
            preprocessor = self.preprocessors[modality]
            
            # Handle multi-slice images
            if img.ndim == 3:
                # Process each slice independently
                preprocessed_slices = []
                for i in range(img.shape[-1]):
                    slice_2d = img[:, :, i]
                    result = preprocessor.preprocess(slice_2d)
                    preprocessed_slices.append(result['preprocessed'])
                return np.stack(preprocessed_slices, axis=-1)
            else:
                # Single 2D image
                result = preprocessor.preprocess(img)
                return result['preprocessed']
                
        except Exception as e:
            print(f"Warning: Preprocessing failed for {modality}: {e}. Using original image.")
            return img


def load_error_series_ids(yaml_path: str) -> set:
    """
    Load erroneous SeriesInstanceUIDs from error_data.yaml.
    
    Args:
        yaml_path: Path to error_data.yaml file
        
    Returns:
        Set of error SeriesInstanceUIDs to exclude
    """
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            return set(data)
        else:
            print(f"Warning: error_data.yaml format unexpected")
            return set()
            
    except FileNotFoundError:
        print(f"Warning: Error data file not found: {yaml_path}")
        return set()
    except Exception as e:
        print(f"Warning: Error loading error_data.yaml: {e}")
        return set()


def create_data_splits(
    train_csv_path: str,
    location_labels: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    error_yaml_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train, validation, and test splits from train_csv_path.
    Optionally filters out error series via error_yaml_path.
    """
    print(f"Loading full dataset from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    print(f"Total series: {len(df)}")

    # Filter out error series if error_yaml provided
    if error_yaml_path and os.path.exists(error_yaml_path):
        error_series_ids = load_error_series_ids(error_yaml_path)
        if error_series_ids:
            original_count = len(df)
            df = df[~df['SeriesInstanceUID'].isin(error_series_ids)]
            removed_count = original_count - len(df)
            print(f"Filtered out {removed_count} error series")
            print(f"Remaining series: {len(df)}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    print(f"\nData splits created:")
    print(f"  Training: {len(train_df)} samples ({len(train_df) / n_total * 100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df) / n_total * 100:.1f}%)")
    print(f"  Testing: {len(test_df)} samples ({len(test_df) / n_total * 100:.1f}%)")
    
    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    modality_dirs: Dict[str, str],
    location_labels: List[str],
    segmentation_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    num_slices: int = 7,
    train_transform=None,
    val_transform=None,
    apply_preprocessing: bool = False,
    preprocessing_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        train_df, val_df, test_df: DataFrames with sample information
        modality_dirs: Dictionary mapping modality names to directories
        location_labels: List of location label names
        segmentation_dir: Directory with segmentation masks
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Target image size
        num_slices: Number of slices for 2.5D
        train_transform: Training augmentation transforms
        val_transform: Validation transforms
        apply_preprocessing: Whether to apply preprocessing (N4, HU windowing, vesselness, etc.)
        preprocessing_config: Dictionary with preprocessing parameters
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = AneurysmDataset(
        data_info=train_df,
        modality_dirs=modality_dirs,
        location_labels=location_labels,
        segmentation_dir=segmentation_dir,
        target_size=target_size,
        num_slices=num_slices,
        transform=train_transform,
        is_training=True,
        apply_preprocessing=apply_preprocessing,
        preprocessing_config=preprocessing_config
    )
    
    val_dataset = AneurysmDataset(
        data_info=val_df,
        modality_dirs=modality_dirs,
        location_labels=location_labels,
        segmentation_dir=segmentation_dir,
        target_size=target_size,
        num_slices=num_slices,
        transform=val_transform,
        is_training=False,
        apply_preprocessing=apply_preprocessing,
        preprocessing_config=preprocessing_config
    )
    
    test_dataset = AneurysmDataset(
        data_info=test_df,
        modality_dirs=modality_dirs,
        location_labels=location_labels,
        segmentation_dir=segmentation_dir,
        target_size=target_size,
        num_slices=num_slices,
        transform=val_transform,
        is_training=False,
        apply_preprocessing=apply_preprocessing,
        preprocessing_config=preprocessing_config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Test dataset loading
    print("Testing AneurysmDataset...")
    
    # Sample configuration
    train_csv = r"D:\FYP_Data\rsna-intracranial-aneurysm-detection\train.csv"
    modality_dirs = {
        'CTA': r"E:\Education\FYP\Preprocessed_images_2.5D\CTA",
        'MRA': r"E:\Education\FYP\Preprocessed_images_2.5D\MRA"
    }
    
    location_labels = [
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
    
    # Create splits
    train_df, val_df, test_df = create_data_splits(
        train_csv_path=train_csv,
        location_labels=location_labels
    )
    
    print(f"\n Data splits created successfully!")
