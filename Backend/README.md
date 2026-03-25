# Intracranial Aneurysm Detection, Localization, and Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning pipeline for multi-task intracranial aneurysm analysis using multi-modal medical imaging (CTA, MRA, MRI). The system performs:

1. **Binary Detection**: Aneurysm present or not
2. **Location Classification**: 13 specific anatomical locations
3. **Segmentation**: Pixel-level aneurysm delineation

## 🌟 Key Features

- **Attention-Based Late Fusion**: Combines predictions from separately trained modality-specific models (CTA, MRA, MRI)
- **Multi-Task Learning**: Simultaneous detection, localization, and segmentation using ResNet50
- **Query-Key-Value Attention**: Learns adaptive feature weighting for each modality
- **2.5D Architecture**: 7-slice representation for spatial context
- **Frozen Base Models**: Only trains attention and fusion layers (~1.73M parameters)
- **Comprehensive Evaluation**: AUC, confusion matrices, location-specific metrics, and visualization

---

## 📁 Project Structure

```
Aerux_Final/
├── config/
│   └── config.py                    # Configuration (13 location labels, paths, hyperparameters)
├── data/
│   ├── dataset.py                   # Multi-modal dataset loader with collate_fn
│   ├── preprocessing.py             # DICOM preprocessing utilities
│   └── select_dataset.py            # Dataset selection and balancing
├── data_splits/
│   └── selected_dataset_2000.csv    # Balanced dataset (1000 CTA, 500 MRA, 500 MRI)
├── Preprocessed_images_2.5D/
│   ├── CTA/                         # Preprocessed CTA volumes (256x256x7)
│   ├── MRA/                         # Preprocessed MRA volumes
│   └── MRI/                         # Preprocessed MRI volumes (T1post + T2)
├── outputs/
│   ├── multitask_CTA_*/             # Individual CTA model checkpoints
│   ├── multitask_MRA_*/             # Individual MRA model checkpoints
│   ├── multitask_MRI_*/             # Individual MRI model checkpoints
│   └── fusion_*/                    # Fusion model checkpoints and results
├── utils/
│   └── visualization.py             # Visualization utilities
├── train_multitask.py               # Train individual modality models (CTA/MRA/MRI)
├── train_fusion.py                  # Train attention-based fusion model
├── inference_fusion.py              # Inference with fusion model
├── visualize_simple.py              # Simple 3-panel visualization (Original|Predicted|GT)
├── visualize_predictions.py         # Detailed prediction visualizations
├── run_fusion_training.py           # Helper: Auto-detect models and train fusion
├── run_inference.py                 # Helper: Auto-detect fusion model and run inference
├── run_visualization.py             # Helper: Auto-detect results and visualize
├── check_data_quality.py            # Diagnostic: Check dataset quality
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project
cd E:\Education\Aerux_Final

# Activate virtual environment
.\env\Scripts\Activate.ps1

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn tqdm nibabel scipy
```

### 2. Dataset Organization

Ensure your dataset follows this structure:

```
F:\FYP_Data\rsna-intracranial-aneurysm-detection\
├── train.csv                  # Contains 13 location labels + metadata
├── train_localizers.csv       # Bounding box coordinates (not used)
├── series/                    # DICOM files (not directly used)
└── segmentations/             # Ground truth masks (NIfTI format)
    ├── {SeriesUID}.nii        # Original segmentation
    └── {SeriesUID}_cowseg.nii # Preprocessed segmentation

E:\Education\Aerux_Final\Preprocessed_images_2.5D\
├── CTA/
│   ├── aneurysm/              # {SeriesUID}.npy (256, 256, 7)
│   └── no_aneurysm/           # {SeriesUID}.npy (256, 256, 7)
├── MRA/
│   ├── aneurysm/
│   └── no_aneurysm/
└── MRI/
    ├── aneurysm/              # T1post + T2 combined
    └── no_aneurysm/

E:\Education\Aerux_Final\data_splits\
└── selected_dataset_2000.csv  # Balanced dataset manifest
```

**Preprocessing:** Images already preprocessed to 2.5D volumes (7 slices, 256×256)

### 3. Preprocessing Techniques (Optional)

The pipeline includes advanced preprocessing techniques from the original research:

#### MRI/MRA Preprocessing Pipeline
1. **N4 Bias Field Correction** - Removes intensity non-uniformity caused by magnetic field inhomogeneities
2. **Sato Vesselness Filter** - Enhances tubular structures (blood vessels) using Hessian-based filtering
3. **Otsu Thresholding** (optional) - Automatic binary segmentation

#### CTA Preprocessing Pipeline
1. **HU (Hounsfield Unit) Windowing** - Focuses on specific tissue density range (default: 40±40 HU for brain/vessels)
2. **CLAHE** (optional) - Contrast Limited Adaptive Histogram Equalization for local contrast enhancement
3. **Sato Vesselness Filter** - Vessel enhancement after HU windowing

#### Enabling Preprocessing

**Via Configuration File (config.yaml):**
```yaml
data:
  # Enable preprocessing
  apply_preprocessing: true
  
  # MRI/MRA settings
  mri_apply_n4: true
  mri_apply_vesselness: true
  mri_apply_otsu: false
  mri_vesselness_scale_range: [1, 8]
  
  # CTA settings
  cta_apply_hu_windowing: true
  cta_window_center: 40.0
  cta_window_width: 80.0
  cta_apply_clahe: false  # Can cause artifacts
  cta_apply_vesselness: true
```

**Via Python Code:**
```python
from data.preprocessing import create_preprocessor

# Create MRI preprocessor
mri_prep = create_preprocessor('MRI', 
                               apply_n4=True,
                               apply_vesselness=True,
                               apply_otsu=False)

# Apply preprocessing
result = mri_prep.preprocess(image)
preprocessed_image = result['preprocessed']

# Create CTA preprocessor
cta_prep = create_preprocessor('CTA',
                               apply_hu_windowing=True,
                               window_center=40,
                               window_width=80,
                               apply_vesselness=True)

result = cta_prep.preprocess(image)
preprocessed_image = result['preprocessed']
```

**Note:** Install required packages for preprocessing:
```bash
pip install SimpleITK scikit-image
```

These preprocessing techniques are applied **on-the-fly during data loading** if enabled in the configuration. They are **optional** and the pipeline works with pre-processed images as well.

### 5. Training Pipeline

#### Step 1: Train Individual Modality Models

Train each modality separately on its specific data:

```bash
# Train CTA model (1000 samples)
python train_multitask.py --modality CTA --epochs 50 --batch_size 8

# Train MRA model (500 samples)
python train_multitask.py --modality MRA --epochs 50 --batch_size 8

# Train MRI model (500 samples - T1post + T2)
python train_multitask.py --modality MRI --epochs 50 --batch_size 8
```

**Output:** Saves best models in `outputs/multitask_{MODALITY}_YYYYMMDD_HHMMSS/`

#### Step 2: Train Attention-Based Fusion Model

Combines all three trained models using attention mechanism:

```bash
# Automatic: Finds latest models and trains fusion
python run_fusion_training.py

# Manual: Specify model paths
python train_fusion.py \
    --cta_model outputs/multitask_CTA_*/best_model_CTA.pth \
    --mra_model outputs/multitask_MRA_*/best_model_MRA.pth \
    --mri_model outputs/multitask_MRI_*/best_model_MRI.pth \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4
```

**What it trains:**
- Attention layers (Q, K, V transformations): 1.6M parameters
- Task-specific fusion heads: 130K parameters
- Base models frozen: 30M+ parameters (no training)

**Output:** Saves fusion model in `outputs/fusion_YYYYMMDD_HHMMSS/`

### 6. Inference & Visualization

```bash
# Run inference with fusion model (automatic detection)
python run_inference.py

# Or specify model manually
python inference_fusion.py --fusion_model outputs/fusion_*/best_fusion_model.pth

# Visualize predictions (simple 3-panel view)
python visualize_simple.py --predictions_dir outputs/fusion_YYYYMMDD_HHMMSS

# Detailed visualizations
python visualize_predictions.py --predictions_dir outputs/fusion_YYYYMMDD_HHMMSS --save_all
```

---

## 📊 13 Aneurysm Location Labels

The model performs multi-label classification for these anatomical locations:

1. Anterior Communicating Artery
2. Left Anterior Cerebral Artery
3. Right Anterior Cerebral Artery
4. Basilar Tip
5. Left Internal Carotid Artery
6. Right Internal Carotid Artery
7. Left Middle Cerebral Artery
8. Right Middle Cerebral Artery
9. Pericallosal Artery
10. Left Posterior Cerebral Artery
11. Right Posterior Cerebral Artery
12. Left Posterior Communicating Artery
13. Right Posterior Communicating Artery

**Output:** Multi-label predictions (patient can have multiple aneurysms)

---

## ⚙️ Configuration

The `config/config.py` file contains all hyperparameters. Key sections:

### Data Configuration
- Dataset paths (CSV files, preprocessed images)
- Train/val/test split ratios
- Image preprocessing parameters (size, normalization)
- 12 location labels

### Model Configuration
- Backbone architecture (ResNet50)
- Multi-task head dimensions
- Attention mechanism settings
- Dropout rates

### Training Configuration
- Optimizer (Adam/AdamW/SGD)
- Learning rate & scheduler
- Loss weights (detection: 1.0, localization: 1.0, segmentation: 2.0)
- Batch size, epochs, early stopping
- Mixed precision training (AMP)

### Example YAML Configuration

```yaml
data:
  train_csv: "D:/FYP_Data/rsna-intracranial-aneurysm-detection/train.csv"
  preprocessed_cta_dir: "E:/Education/FYP/Preprocessed_images_2.5D/CTA"
  target_size: [256, 256]
  num_slices: 7

model:
  backbone: "resnet50"
  num_detection_classes: 2
  num_location_classes: 13
  enable_segmentation: true

training:
  num_epochs: 100
  batch_size: 8
  learning_rate: 0.0001
  detection_loss_weight: 1.0
  localization_loss_weight: 1.0
  segmentation_loss_weight: 2.0
```

---

## 🧠 Model Architecture

### Phase 1: Individual Modality Models (MultiTaskResNet50)

Each modality (CTA, MRA, MRI) has its own trained model:

**Encoder (ResNet50):**
- Modified first conv layer: 7-channel input (2.5D slices)
- Pretrained ImageNet weights adapted for medical imaging
- Layers: conv1 → layer1 (256) → layer2 (512) → layer3 (1024) → layer4 (2048)

**Multi-Task Heads:**
1. **Detection Head**: Global pooling → FC(2048→512→2) → Binary classification
2. **Location Head**: Global pooling → FC(2048→512→13) → Multi-label (13 locations)
3. **Segmentation Decoder**: U-Net style with skip connections → 256×256 mask

### Phase 2: Attention-Based Late Fusion

**Architecture:**
```
Input (CTA/MRA/MRI) → Frozen Encoder → Features (2048D)
                                           ↓
                          [Detach + Enable Gradients]
                                           ↓
                          Attention Mechanism (Q, K, V)
                          ↓         ↓         ↓
                        256D      256D      256D
                                  ↓
                          Scaled Dot-Product
                                  ↓
                          Attended Features (256D)
                          ↓         ↓         ↓
                    Detection   Location  Segmentation
                       Head       Head      Decoder
```

**Trainable Components:**
- Query/Key/Value projections: `Linear(2048 → 256)` × 3
- Detection fusion: `256 → 128 → 2`
- Location fusion: `256 → 128 → 64 → 13`
- Segmentation fusion: `256 → 128 → 64` + `ConvTranspose` upsampling

**Key Features:**
- **Frozen base models**: No retraining of 30M+ parameters
- **Attention learning**: Adapts to modality-specific features
- **Modality routing**: Each sample uses its native modality
- **Gradient flow**: `features.detach().requires_grad = True`

---

## 📈 Training Details

### Phase 1: Individual Modality Training

**Loss Functions:**
1. **Detection**: CrossEntropyLoss (binary)
2. **Location**: BCEWithLogitsLoss (multi-label, 13 locations)
3. **Segmentation**: Dice + BCE combined loss

**Combined Loss:**
```
L_total = w_det * L_det + w_loc * L_loc + w_seg * L_seg
Default weights: detection=1.0, localization=1.0, segmentation=2.0
```

**Data Augmentation:**
- Random horizontal flip (50%)
- Random vertical flip (50%)
- Random rotation (90°, 180°, 270°)

**Optimization:**
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Mixed Precision: Enabled (torch.amp)
- Early Stopping: Patience=10 epochs

**Dataset:**
- CTA: 1000 samples (700 train, 150 val, 150 test)
- MRA: 500 samples (350 train, 75 val, 75 test)
- MRI: 500 samples (350 train, 75 val, 75 test)

### Phase 2: Fusion Training

**Loss Configuration:**
```
detection_weight = 1.0
localization_weight = 2.5  # Increased for better location learning
segmentation_weight = 0.5   # Reduced (0% mask coverage)
```

**Optimization:**
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Only trains attention + fusion layers (~1.73M params)
- Base models frozen (30M+ params)
- Epochs: 30
- Batch size: 8

**Dataset:**
- Combined: 1952 samples (all modalities)
- Train/Val/Test: 70%/15%/15% split
- Custom collate_fn: Handles variable modality presence

---

## 📊 Evaluation Metrics

### Detection Metrics
- **AUC-ROC**: Area under ROC curve
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: 2×2 matrix

### Location Metrics
- **Accuracy**: Multi-class classification accuracy
- **Per-Class Precision/Recall**: For each of 12 locations
- **Confusion Matrix**: 13×13 matrix

### Segmentation Metrics
- **Dice Coefficient**: Overlap measure (mean ± std)
- **IoU (Intersection over Union)**: Jaccard index
- **Pixel Accuracy**: Correct pixels / total pixels

---

## 🎨 Visualization Tools

The pipeline generates comprehensive visualizations:

1. **Training History**: Loss curves, accuracy plots
2. **ROC Curve**: Detection performance
3. **Confusion Matrices**: Detection and location classification
4. **Attention Maps**: Spatial attention heatmaps
5. **Segmentation Overlays**: Predicted vs ground truth masks
6. **Location Distribution**: Class distribution analysis

Example:
```python
from utils.visualization import create_visualization_report

create_visualization_report(
    metrics=eval_metrics,
    train_history=train_history,
    val_history=val_history,
    results=eval_results,
    location_labels=config.data.location_labels,
    output_dir='visualizations/'
)
```

---

## 🔬 Advanced Usage

### Custom Configuration

Create a custom `config.yaml`:

```yaml
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

model:
  backbone: "resnet50"
  fusion_dim: 512
  num_attention_heads: 8

training:
  num_epochs: 150
  batch_size: 12
  learning_rate: 0.0001
  use_amp: true
```

Load with:
```bash
python train.py --config my_config.yaml
```

### Fine-Tuning Multi-Modal Models

1. Train single-modal models first:
```bash
python train.py --mode single_modal --modality CTA --num_epochs 50
python train.py --mode single_modal --modality MRA --num_epochs 50
```

2. Train fusion network with frozen encoders:
```bash
python train.py --mode multi_modal --freeze_encoders --num_epochs 100
```

3. Fine-tune end-to-end (optional):
```bash
python train.py --mode multi_modal --num_epochs 50
```

### Ensemble Inference

Combine predictions from multiple models for improved performance:

```python
# Load multiple checkpoints
checkpoints = [
    'outputs/cta_model/best_model.pth',
    'outputs/mra_model/best_model.pth',
    'outputs/fusion_model/best_model.pth'
]

# Average predictions (implement custom logic)
```

---

## 📚 API Reference

### Dataset API

```python
from data.dataset import AneurysmDataset, create_dataloaders

# Create dataset
dataset = AneurysmDataset(
    data_info=train_df,
    modality_dirs={'CTA': 'path/to/cta', 'MRA': 'path/to/mra'},
    location_labels=location_labels,
    target_size=(256, 256),
    num_slices=7
)

# Get sample
sample = dataset[0]
# Returns: {
#   'cta': Tensor(7, 256, 256),
#   'detection_label': int,
#   'location_label': int,
#   'segmentation_mask': Tensor(256, 256)
# }
```

### Model API

```python
from models.resnet_2_5d import MultiTaskResNet2_5D

# Create model
model = MultiTaskResNet2_5D(
    input_channels=7,
    num_detection_classes=2,
    num_location_classes=13,
    num_segmentation_classes=2,
    enable_segmentation=True
)

# Forward pass
outputs = model(input_tensor)
# Returns: {
#   'detection_logits': Tensor(B, 2),
#   'location_logits': Tensor(B, 13),
#   'segmentation_logits': Tensor(B, 2, H, W),
#   'location_attention': Tensor(B, 1, H, W)
# }
```

### Training API

```python
from training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    config=config,
    device=device
)

trainer.train(num_epochs=100)
```

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size
python train.py --batch_size 4

# Enable gradient accumulation
# Edit config.py: accumulation_steps = 8
```

**CUDA Out of Memory**
```bash
# Use mixed precision training (enabled by default)
# Or reduce image size in config.py: target_size = (128, 128)
```

**Low Validation Performance**
- Check class imbalance (use class weights)
- Increase data augmentation
- Reduce learning rate
- Train for more epochs

**Segmentation Metrics are Zero**
- Ensure segmentation masks are available
- Check mask preprocessing (binary values 0/1)
- Increase segmentation loss weight

---

## 📄 Citation

If you use this codebase, please cite:

```bibtex
@software{aneurysm_detection_2025,
  title={Multi-Task Intracranial Aneurysm Detection, Localization, and Segmentation},
  author={Senior Computer Vision Engineer},
  year={2025},
  url={https://github.com/yourrepo/aerux_final}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Submit a pull request

---

## 📧 Contact

For questions or issues:
- **GitHub Issues**: [github.com/yourrepo/issues](https://github.com/yourrepo/issues)
- **Email**: your.email@example.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- RSNA Intracranial Aneurysm Detection Challenge
- PyTorch Team for the deep learning framework
- ResNet authors for the backbone architecture
- Medical imaging community for dataset contributions

---

## 🔄 Version History

- **v1.0.0** (2025-01-08): Initial release
  - Multi-modal fusion with attention mechanisms
  - Multi-task learning (detection, localization, segmentation)
  - 12 location classification
  - Comprehensive evaluation and visualization

---

**Built with ❤️ for medical AI research**
