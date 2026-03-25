"""
Configuration Management Module
Centralizes all configuration parameters for the Intracranial Aneurysm Detection,
Localization, and Segmentation Pipeline.

Author: Senior Computer Vision Engineer
Date: 2025-01-08
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import yaml


@dataclass
class DataConfig:
    """Data-related configuration parameters"""
    
    # Dataset root directory
    dataset_root: str = r"G:\FYP_Data\rsna-intracranial-aneurysm-detection"
    
    # CSV file paths
    train_csv: str = r"G:\FYP_Data\rsna-intracranial-aneurysm-detection\train.csv"
    train_localizers_csv: str = r"G:\FYP_Data\rsna-intracranial-aneurysm-detection\train_localizers.csv"
    
    # Preprocessed data directories
    preprocessed_cta_dir: str = r"E:\Education\Aerux_Final\Preprocessed_images_2.5D\CTA"
    preprocessed_mra_dir: str = r"E:\Education\Aerux_Final\Preprocessed_images_2.5D\MRA"
    preprocessed_mri_dir: str = r"E:\Education\Aerux_Final\Preprocessed_images_2.5D\MRI"
    
    # Segmentation masks directory (if available)
    segmentation_masks_dir: str = r"G:\FYP_Data\rsna-intracranial-aneurysm-detection\segmentations"
    
    # Series (DICOM files) directory
    series_dir: str = r"G:\FYP_Data\rsna-intracranial-aneurysm-detection\series"
    
    # Output directories
    output_root: str = r"E:\Education\Aerux_Final\outputs"
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Image preprocessing parameters
    target_size: Tuple[int, int] = (256, 256)
    num_slices: int = 7  # For 2.5D representation
    normalize_range: Tuple[float, float] = (0.0, 1.0)
    
    # 12 Aneurysm Location Labels (from train.csv)
    location_labels: List[str] = field(default_factory=lambda: [
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
    ])
    
    # Modality types (Simplified: T1post and T2 are both treated as MRI)
    modalities: List[str] = field(default_factory=lambda: ["CTA", "MRA", "MRI"])
    
    # Dataset selection parameters
    use_selected_dataset: bool = False  # Use complete dataset instead of 2000 series subset
    selected_dataset_csv: str = r"E:\Education\Aerux_Final\data_splits\selected_dataset_2000.csv"
    error_data_yaml: str = r"D:\FYP_Data\rsna-intracranial-aneurysm-detection\error_data.yaml"
    
    # Dataset composition (for reference)
    target_cta_count: int = 1000  # 500 aneurysm + 500 no aneurysm
    target_mra_count: int = 500   # 250 aneurysm + 250 no aneurysm
    target_mri_count: int = 500   # 250 aneurysm (125 T1 + 125 T2) + 250 no aneurysm (125 T1 + 125 T2)
    
    # Preprocessing options
    apply_preprocessing: bool = True  # Enable/disable preprocessing during data loading
    
    # MRI/MRA Preprocessing parameters
    mri_apply_n4: bool = True  # N4 Bias Field Correction
    mri_apply_vesselness: bool = True  # Sato vesselness filter
    mri_apply_otsu: bool = False  # Otsu thresholding (optional)
    mri_vesselness_scale_range: Tuple[int, int] = (1, 8)
    mri_vesselness_scale_step: int = 2
    mri_n4_max_iterations: List[int] = field(default_factory=lambda: [50, 50, 50, 50])
    
    # CTA Preprocessing parameters
    cta_apply_hu_windowing: bool = True  # Hounsfield Unit windowing
    cta_apply_clahe: bool = False  # CLAHE enhancement (optional, can cause artifacts)
    cta_apply_vesselness: bool = True  # Sato vesselness filter
    cta_window_center: float = 40.0  # HU window center for brain/vessels
    cta_window_width: float = 80.0  # HU window width
    cta_clahe_clip_limit: float = 3.0  # CLAHE contrast limit
    cta_clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    cta_vesselness_scale_range: Tuple[int, int] = (1, 8)
    cta_vesselness_scale_step: int = 2
    
    def __post_init__(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.output_root, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # Backbone architecture
    backbone: str = "resnet50"  # Options: resnet50, resnet101, efficientnet
    pretrained: bool = True
    
    # Input specifications
    input_channels: int = 7  # For 2.5D representation
    input_size: Tuple[int, int] = (256, 256)
    
    # Multi-task outputs
    num_detection_classes: int = 2  # Binary: aneurysm present or not
    num_location_classes: int = 13  # 12 locations + 1 for no aneurysm
    enable_segmentation: bool = True
    
    # Feature dimensions
    feature_dim: int = 2048  # ResNet50 final feature dimension
    fusion_dim: int = 512
    attention_dim: int = 256
    
    # Dropout rates
    dropout_rate: float = 0.3
    attention_dropout: float = 0.2
    
    # Segmentation decoder
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    segmentation_classes: int = 2  # Binary mask: aneurysm region or background
    
    # Attention mechanism
    attention_type: str = "multi_head"  # Options: simple, multi_head, spatial
    num_attention_heads: int = 8
    
    # Multi-modal fusion
    fusion_strategy: str = "attention_weighted"  # Options: concat, attention_weighted, late_fusion
    freeze_backbone: bool = True  # Freeze backbone during fusion training


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    # Optimizer settings
    optimizer: str = "adam"  # Options: adam, adamw, sgd
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 4  # Gradient accumulation for larger effective batch size
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_loss"  # Options: val_loss, val_auc, val_dice
    
    # Loss weights for multi-task learning
    detection_loss_weight: float = 1.0
    localization_loss_weight: float = 1.0
    segmentation_loss_weight: float = 2.0
    
    # Loss functions
    detection_loss: str = "cross_entropy"  # Options: cross_entropy, focal_loss
    localization_loss: str = "cross_entropy"
    segmentation_loss: str = "dice_bce"  # Dice + BCE combined
    
    # Class weights for imbalanced data
    use_class_weights: bool = True
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Checkpoint settings
    save_frequency: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_frequency: int = 10  # Log every N batches
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class AugmentationConfig:
    """Data augmentation parameters"""
    
    # Spatial augmentations
    random_rotation: bool = True
    rotation_degrees: Tuple[int, int] = (-15, 15)
    
    random_flip: bool = True
    flip_probability: float = 0.5
    
    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    random_translation: bool = True
    translation_range: Tuple[float, float] = (-0.1, 0.1)
    
    # Intensity augmentations
    random_brightness: bool = True
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    
    random_contrast: bool = True
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    random_gamma: bool = True
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise augmentation
    random_noise: bool = True
    noise_std: float = 0.01
    
    # Elastic deformation
    elastic_deformation: bool = False
    elastic_alpha: float = 1.0
    elastic_sigma: float = 0.1
    
    # Augmentation probability
    augmentation_probability: float = 0.8


@dataclass
class InferenceConfig:
    """Inference and evaluation settings"""
    
    # Model checkpoint path
    checkpoint_path: str = ""
    
    # Batch size for inference
    batch_size: int = 16
    
    # Test-time augmentation
    use_tta: bool = False
    tta_transforms: int = 4
    
    # Ensemble settings
    use_ensemble: bool = False
    ensemble_weights: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Thresholds
    detection_threshold: float = 0.5
    segmentation_threshold: float = 0.5
    
    # Output settings
    save_predictions: bool = True
    save_visualizations: bool = True
    visualization_dir: str = "visualizations"
    
    # Metrics to compute
    compute_detection_metrics: bool = True
    compute_localization_metrics: bool = True
    compute_segmentation_metrics: bool = True


class Config:
    """
    Main configuration class that combines all sub-configurations
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file or use defaults
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.augmentation = AugmentationConfig()
        self.inference = InferenceConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if 'augmentation' in config_dict:
            for key, value in config_dict['augmentation'].items():
                if hasattr(self.augmentation, key):
                    setattr(self.augmentation, key, value)
        
        if 'inference' in config_dict:
            for key, value in config_dict['inference'].items():
                if hasattr(self.inference, key):
                    setattr(self.inference, key, value)
    
    def save_to_yaml(self, save_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'inference': self.inference.__dict__
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_location_label_mapping(self) -> Dict[int, str]:
        """Get mapping from index to location label"""
        mapping = {i: label for i, label in enumerate(self.data.location_labels)}
        mapping[len(self.data.location_labels)] = "No Aneurysm"
        return mapping
    
    def get_modality_mapping(self) -> Dict[int, str]:
        """Get mapping from index to modality"""
        return {i: modality for i, modality in enumerate(self.data.modalities)}
    
    def create_output_directories(self):
        """Create all necessary output directories"""
        dirs_to_create = [
            os.path.join(self.data.output_root, 'checkpoints'),
            os.path.join(self.data.output_root, 'logs'),
            os.path.join(self.data.output_root, 'visualizations'),
            os.path.join(self.data.output_root, 'predictions'),
            os.path.join(self.data.output_root, 'tensorboard'),
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"Config(\n"
            f"  Data: {self.data}\n"
            f"  Model: {self.model}\n"
            f"  Training: {self.training}\n"
            f"  Augmentation: {self.augmentation}\n"
            f"  Inference: {self.inference}\n"
            f")"
        )


# Create a default configuration instance
def get_default_config() -> Config:
    """Get default configuration instance"""
    return Config()


# Example usage
if __name__ == "__main__":
    # Create default config
    config = get_default_config()
    
    # Print configuration
    print("=" * 80)
    print("INTRACRANIAL ANEURYSM DETECTION, LOCALIZATION & SEGMENTATION")
    print("Configuration Summary")
    print("=" * 80)
    print(f"\nLocation Labels ({len(config.data.location_labels)}):")
    for i, label in enumerate(config.data.location_labels, 1):
        print(f"  {i:2d}. {label}")
    
    print(f"\nModalities ({len(config.data.modalities)}):")
    for i, modality in enumerate(config.data.modalities, 1):
        print(f"  {i}. {modality}")
    
    print(f"\nModel Configuration:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Input Size: {config.model.input_size}")
    print(f"  Detection Classes: {config.model.num_detection_classes}")
    print(f"  Location Classes: {config.model.num_location_classes}")
    print(f"  Segmentation Enabled: {config.model.enable_segmentation}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Mixed Precision: {config.training.use_amp}")
    
    # Create output directories
    config.create_output_directories()
    
    # Save configuration to YAML
    yaml_path = os.path.join(config.data.output_root, 'config.yaml')
    config.save_to_yaml(yaml_path)
    print(f"\n Configuration saved to: {yaml_path}")
    print("=" * 80)
