"""
Multi-Modal ATTENTION-BASED FUSION Training Script

This script implements attention-based fusion for combining 
three separately trained models (CTA, MRA, MRI).

WHAT IS ATTENTION-BASED FUSION?
- Each sample (CTA/MRA/MRI) is processed by its respective pre-trained model
- Pre-trained models are FROZEN (weights don't change)
- Learnable ATTENTION MECHANISM adaptively weights features from each model
- Attention learns which aspects of each modality are most important

TRAINING PROCESS:
1. Load 3 pre-trained models (CTA trained on 1000 images, MRA on 500, MRI on 500)
2. Freeze all model weights
3. Add attention mechanism (query, key, value transformations)
4. Train attention layers on combined dataset (2000 samples)
5. Each sample's features are weighted by learned attention scores

ADVANTAGES:
- Leverages already trained models (no retraining base models)
- Adaptive feature weighting (more powerful than fixed calibration)
- Learns modality-specific importance dynamically
- Better interpretability through attention weights

Usage:
    python train_fusion.py \
        --cta_model path/to/best_model_CTA.pth \
        --mra_model path/to/best_model_MRA.pth \
        --mri_model path/to/best_model_MRI.pth \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-4
        
Or simply run:
    python run_fusion_training.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Force matplotlib to use non-interactive backend (fixes threading issues)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

sys.path.insert(0, str(Path(__file__).parent))
from config.config import Config
from train_multitask import MultiTaskResNet50, MultiTaskLoss, compute_weighted_auc, LOCATION_LABELS


def multimodal_collate_fn(batch):
    """Custom collate function to handle None values in multi-modal data"""
    # Separate batch components
    inputs_list = [item[0] for item in batch]
    det_labels_list = [item[1] for item in batch]
    loc_labels_list = [item[2] for item in batch]
    seg_masks_list = [item[3] for item in batch]
    
    # Collate inputs (dict of modality tensors)
    # Track original indices for each modality to reorder predictions later
    modalities = ['CTA', 'MRA', 'MRI']
    
    collated_inputs = {}
    modality_indices = {}  # Maps modality -> list of original batch indices
    valid_sample_indices = set()  # Track which samples have at least one valid modality
    
    for modality in modalities:
        # Collect all non-None tensors for this modality and their indices
        modality_tensors = []
        indices = []
        
        for i, inp in enumerate(inputs_list):
            tensor = inp.get(modality)
            # Check if tensor is valid (not None and has correct shape)
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                modality_tensors.append(tensor)
                indices.append(i)
                valid_sample_indices.add(i)
        
        if len(modality_tensors) > 0:
            # Stack available tensors
            collated_inputs[modality] = torch.stack(modality_tensors)
            modality_indices[modality] = indices
            # Validate consistency
            assert collated_inputs[modality].shape[0] == len(indices), \
                f"Mismatch in {modality}: tensor batch {collated_inputs[modality].shape[0]} vs indices {len(indices)}"
        else:
            # No samples of this modality in batch
            collated_inputs[modality] = None
            modality_indices[modality] = []
    
    # Filter labels to only include samples with valid modality data
    # Create mapping from old indices to new indices
    valid_indices_sorted = sorted(list(valid_sample_indices))
    
    if len(valid_indices_sorted) == 0:
        # No valid samples in batch - return empty batch
        return collated_inputs, torch.zeros(0, dtype=torch.long), torch.zeros(0, 13), torch.zeros(0, 1, 256, 256), {}
    
    # Remap modality_indices to new condensed indices
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices_sorted)}
    
    for modality in modalities:
        if modality_indices[modality]:
            modality_indices[modality] = [old_to_new_idx[old_idx] for old_idx in modality_indices[modality]]
    
    # Filter labels
    det_labels = torch.stack([det_labels_list[i] for i in valid_indices_sorted])
    loc_labels = torch.stack([loc_labels_list[i] for i in valid_indices_sorted])
    seg_masks = torch.stack([seg_masks_list[i] for i in valid_indices_sorted])
    
    return collated_inputs, det_labels, loc_labels, seg_masks, modality_indices


class MultiModalDataset(Dataset):
    """Dataset for multi-modal fusion training"""
    
    def __init__(self, dataframe, base_dirs, train_csv_path, segmentation_dir=None, augment=False):
        """
        Args:
            dataframe: DataFrame with SeriesInstanceUID and Modality
            base_dirs: Dict mapping modality to base directory
            train_csv_path: Path to train.csv
            segmentation_dir: Directory with segmentation masks
            augment: Apply data augmentation
        """
        self.df = dataframe
        self.base_dirs = base_dirs
        self.segmentation_dir = segmentation_dir
        self.augment = augment
        
        # Load train.csv for location labels
        self.train_csv = pd.read_csv(train_csv_path)
    
    def __len__(self):
        return len(self.df)
    
    def _get_location_labels(self, series_uid):
        """Get location labels for a series"""
        series_row = self.train_csv[self.train_csv['SeriesInstanceUID'] == series_uid]
        location_vector = np.zeros(13, dtype=np.float32)
        
        if len(series_row) == 0:
            return location_vector
        
        row = series_row.iloc[0]
        if pd.isna(row.get('Aneurysm Present', 0)) or row.get('Aneurysm Present', 0) == 0:
            return location_vector
        
        for i, label in enumerate(LOCATION_LABELS):
            if label in row and pd.notna(row[label]) and row[label] == 1:
                location_vector[i] = 1.0
        
        return location_vector
    
    def _load_volume(self, series_uid, modality, detection_label):
        """Load a single modality volume"""
        base_dir = self.base_dirs.get(modality)
        if base_dir is None:
            return None
        
        label_name = 'aneurysm' if detection_label == 1 else 'no_aneurysm'
        file_path = os.path.join(base_dir, label_name, f"{series_uid}.npy")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            volume = np.load(file_path).astype(np.float32)
            volume = np.transpose(volume, (2, 0, 1))  # (7, 256, 256)
            return volume
        except:
            return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        detection_label = int(row['Aneurysm Present'])
        modality = row['Modality']
        
        # Determine which modality this series belongs to
        if 'MRI' in modality.upper():
            primary_modality = 'MRI'
        else:
            primary_modality = modality.upper()
        
        # For fusion training, we only load the primary modality
        # (Each series exists in only one modality)
        volumes = {
            'CTA': None,
            'MRA': None,
            'MRI': None
        }
        
        # Load only the primary modality
        vol = self._load_volume(series_uid, primary_modality, detection_label)
        volumes[primary_modality] = vol
        
        # Get location labels
        location_labels = self._get_location_labels(series_uid)
        
        # Load segmentation mask
        if self.segmentation_dir and detection_label == 1:
            mask_path = os.path.join(self.segmentation_dir, f"{series_uid}.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(np.float32)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
            else:
                mask = np.zeros((256, 256), dtype=np.float32)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment:
            volumes, mask = self._augment(volumes, mask)
        
        # Convert to tensors
        volume_tensors = {}
        for mod, vol in volumes.items():
            if vol is not None:
                volume_tensors[mod] = torch.from_numpy(vol)
            else:
                volume_tensors[mod] = None
        
        return (
            volume_tensors,
            torch.tensor(detection_label, dtype=torch.long),
            torch.from_numpy(location_labels),
            torch.from_numpy(mask).unsqueeze(0)
        )
    
    def _augment(self, volumes, mask):
        """Data augmentation applied consistently across all modalities"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            volumes = {k: np.flip(v, axis=2).copy() if v is not None else None 
                      for k, v in volumes.items()}
            mask = np.flip(mask, axis=1).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            volumes = {k: np.flip(v, axis=1).copy() if v is not None else None 
                      for k, v in volumes.items()}
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            volumes = {mod: np.rot90(v, k, axes=(1, 2)).copy() if v is not None else None 
                      for mod, v in volumes.items()}
            mask = np.rot90(mask, k).copy()
        
        return volumes, mask


class SimpleFusionModel(nn.Module):
    """Attention-based fusion model that combines predictions from multiple modality models"""
    
    def __init__(self, modality_models: Dict[str, nn.Module], fusion_strategy='attention'):
        """
        Args:
            modality_models: Dict of {modality: trained_model}
            fusion_strategy: 'attention' (recommended) or 'average'
        """
        super(SimpleFusionModel, self).__init__()
        
        self.modality_models = nn.ModuleDict(modality_models)
        self.fusion_strategy = fusion_strategy
        self.modalities = list(modality_models.keys())
        
        # Attention-based fusion components (ALWAYS create these for trainable parameters)
        # Feature dimension from ResNet50 backbone (before task heads)
        feature_dim = 2048
        
        # Multi-head attention for learning modality importance
        self.attention_query = nn.Linear(feature_dim, 256)
        self.attention_key = nn.Linear(feature_dim, 256)
        self.attention_value = nn.Linear(feature_dim, 256)
        
        # Attention combination layers for each task
        self.detection_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        self.location_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 13)
        )
        
        self.segmentation_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Final segmentation upsampling
        self.seg_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )
        
        # Freeze individual modality models AFTER creating attention layers
        for model in self.modality_models.values():
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, inputs: Dict[str, Optional[torch.Tensor]], modality_indices: Dict[str, list]):
        """
        Args:
            inputs: Dict of {modality: tensor or None}
                   Each modality tensor contains only samples of that modality
            modality_indices: Dict of {modality: [original_batch_indices]}
        Returns:
            Fused predictions using attention mechanism
        """
        # Get batch size from indices
        batch_size = sum(len(indices) for indices in modality_indices.values())
        
        if batch_size == 0:
            print("WARNING: batch_size is 0! Returning dummy tensors.")
            device = next(self.parameters()).device
            return (
                torch.zeros(1, 2, device=device),
                torch.zeros(1, 13, device=device),
                torch.zeros(1, 1, 256, 256, device=device)
            )
        
        device = next(self.parameters()).device
        
        # Initialize output tensors
        det_logits_out = torch.zeros(batch_size, 2, device=device)
        loc_logits_out = torch.zeros(batch_size, 13, device=device)
        seg_pred_out = torch.zeros(batch_size, 1, 256, 256, device=device)
        
        # Process each modality with attention-based fusion
        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                modality_tensor = inputs[modality]
                indices = modality_indices.get(modality, [])
                
                # Validate consistency
                tensor_batch_size = modality_tensor.shape[0]
                if len(indices) != tensor_batch_size:
                    print(f"Warning: {modality} indices mismatch")
                    indices = indices[:tensor_batch_size]
                    if len(indices) < tensor_batch_size:
                        continue
                
                # Extract features from frozen encoder (MultiTaskResNet50 architecture)
                model = self.modality_models[modality]
                with torch.no_grad():
                    # Forward through encoder layers to get 2048-dim features
                    x = model.conv1(modality_tensor)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)  # [B, 2048, 8, 8]
                    # Global average pooling
                    features = model.avgpool(x)  # [B, 2048, 1, 1]
                    features = torch.flatten(features, 1)  # [B, 2048]
                
                # Detach and enable gradients for attention mechanism
                features = features.detach()
                features.requires_grad = True
                
                # Apply attention mechanism (these layers are trainable)
                queries = self.attention_query(features)  # [B, 256]
                keys = self.attention_key(features)  # [B, 256]
                values = self.attention_value(features)  # [B, 256]
                
                # Compute attention scores (scaled dot-product attention)
                attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (256 ** 0.5)
                attention_weights = F.softmax(attention_scores, dim=-1)
                attended_features = torch.matmul(attention_weights, values)  # [B, 256]
                
                # Generate task-specific predictions from attended features
                det_logits = self.detection_fusion(attended_features)
                loc_logits = self.location_fusion(attended_features)
                
                # Segmentation: reshape features for spatial prediction
                seg_features = self.segmentation_fusion(attended_features)  # [B, 64]
                seg_features = seg_features.view(-1, 64, 1, 1)  # [B, 64, 1, 1]
                seg_features = F.interpolate(seg_features, size=(32, 32), mode='bilinear', align_corners=False)
                seg_pred = self.seg_upsample(seg_features)  # [B, 1, 256, 256]
                
                # Place predictions at original batch positions
                for i, orig_idx in enumerate(indices):
                    if i < det_logits.shape[0] and orig_idx < batch_size:
                        det_logits_out[orig_idx] = det_logits[i]
                        loc_logits_out[orig_idx] = loc_logits[i]
                        seg_pred_out[orig_idx] = seg_pred[i]
        
        return det_logits_out, loc_logits_out, seg_pred_out


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_det_loss = 0.0
    running_loc_loss = 0.0
    running_seg_loss = 0.0
    
    all_det_preds = []
    all_det_labels = []
    all_det_probs = []
    all_loc_probs = []
    all_loc_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (inputs, det_labels, loc_labels, seg_masks, modality_indices) in enumerate(progress_bar):
        # Skip empty batches (no valid samples)
        if det_labels.shape[0] == 0:
            continue
        
        # Move inputs to device
        inputs_device = {}
        for mod, vol in inputs.items():
            if vol is not None:
                inputs_device[mod] = vol.to(device)
            else:
                inputs_device[mod] = None
        
        det_labels = det_labels.to(device)
        loc_labels = loc_labels.to(device)
        seg_masks = seg_masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast('cuda'):
                det_logits, loc_logits, seg_pred = model(inputs_device, modality_indices)
                total_loss, det_loss, loc_loss, seg_loss = criterion(
                    det_logits, loc_logits, seg_pred,
                    det_labels, loc_labels, seg_masks
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            det_logits, loc_logits, seg_pred = model(inputs_device, modality_indices)
            total_loss, det_loss, loc_loss, seg_loss = criterion(
                det_logits, loc_logits, seg_pred,
                det_labels, loc_labels, seg_masks
            )
            total_loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_det_loss += det_loss.item()
        running_loc_loss += loc_loss.item()
        running_seg_loss += seg_loss.item()
        
        det_probs = torch.softmax(det_logits, dim=1)[:, 1].detach().cpu().numpy()
        det_preds = torch.argmax(det_logits, dim=1).detach().cpu().numpy()
        loc_probs = torch.sigmoid(loc_logits).detach().cpu().numpy()
        
        all_det_preds.extend(det_preds)
        all_det_labels.extend(det_labels.cpu().numpy())
        all_det_probs.extend(det_probs)
        all_loc_probs.append(loc_probs)
        all_loc_labels.append(loc_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': total_loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    det_loss = running_det_loss / len(dataloader)
    loc_loss = running_loc_loss / len(dataloader)
    seg_loss = running_seg_loss / len(dataloader)
    
    all_loc_probs = np.vstack(all_loc_probs)
    all_loc_labels = np.vstack(all_loc_labels)
    
    weighted_auc, det_auc, loc_aucs = compute_weighted_auc(
        all_det_probs, all_det_labels, all_loc_probs, all_loc_labels
    )
    
    return epoch_loss, det_loss, loc_loss, seg_loss, weighted_auc, det_auc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    running_det_loss = 0.0
    running_loc_loss = 0.0
    running_seg_loss = 0.0
    
    all_det_preds = []
    all_det_labels = []
    all_det_probs = []
    all_loc_probs = []
    all_loc_labels = []
    
    with torch.no_grad():
        for inputs, det_labels, loc_labels, seg_masks, modality_indices in tqdm(dataloader, desc='Validating'):
            # Skip empty batches (no valid samples)
            if det_labels.shape[0] == 0:
                continue
            
            # Move inputs to device
            inputs_device = {}
            for mod, vol in inputs.items():
                if vol is not None:
                    inputs_device[mod] = vol.to(device)
                else:
                    inputs_device[mod] = None
            
            det_labels = det_labels.to(device)
            loc_labels = loc_labels.to(device)
            seg_masks = seg_masks.to(device)
            
            det_logits, loc_logits, seg_pred = model(inputs_device, modality_indices)
            total_loss, det_loss, loc_loss, seg_loss = criterion(
                det_logits, loc_logits, seg_pred,
                det_labels, loc_labels, seg_masks
            )
            
            running_loss += total_loss.item()
            running_det_loss += det_loss.item()
            running_loc_loss += loc_loss.item()
            running_seg_loss += seg_loss.item()
            
            det_probs = torch.softmax(det_logits, dim=1)[:, 1].cpu().numpy()
            det_preds = torch.argmax(det_logits, dim=1).cpu().numpy()
            loc_probs = torch.sigmoid(loc_logits).cpu().numpy()
            
            all_det_preds.extend(det_preds)
            all_det_labels.extend(det_labels.cpu().numpy())
            all_det_probs.extend(det_probs)
            all_loc_probs.append(loc_probs)
            all_loc_labels.append(loc_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    det_loss = running_det_loss / len(dataloader)
    loc_loss = running_loc_loss / len(dataloader)
    seg_loss = running_seg_loss / len(dataloader)
    
    det_accuracy = accuracy_score(all_det_labels, all_det_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_det_labels, all_det_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_det_labels, all_det_preds)
    
    all_loc_probs = np.vstack(all_loc_probs)
    all_loc_labels = np.vstack(all_loc_labels)
    
    weighted_auc, det_auc, loc_aucs = compute_weighted_auc(
        all_det_probs, all_det_labels, all_loc_probs, all_loc_labels
    )
    
    return {
        'loss': epoch_loss,
        'det_loss': det_loss,
        'loc_loss': loc_loss,
        'seg_loss': seg_loss,
        'detection_accuracy': det_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detection_auc': det_auc,
        'weighted_auc': weighted_auc,
        'location_aucs': loc_aucs,
        'confusion_matrix': cm
    }


def plot_training_history(history, save_path):
    """Plot training history for fusion model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Weighted AUC (Competition Metric)
    axes[0, 1].plot(history['train_weighted_auc'], label='Train Weighted AUC', marker='o')
    axes[0, 1].plot(history['val_weighted_auc'], label='Val Weighted AUC', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Weighted AUC')
    axes[0, 1].set_title('Competition Metric: Weighted AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Detection AUC
    axes[0, 2].plot(history['train_det_auc'], label='Train Detection AUC', marker='o')
    axes[0, 2].plot(history['val_det_auc'], label='Val Detection AUC', marker='o')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('Detection AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Component Losses
    axes[1, 0].plot(history['train_det_loss'], label='Detection', marker='o')
    axes[1, 0].plot(history['train_loc_loss'], label='Localization', marker='s')
    axes[1, 0].plot(history['train_seg_loss'], label='Segmentation', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training: Component Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Component Losses
    axes[1, 1].plot(history['val_det_loss'], label='Detection', marker='o')
    axes[1, 1].plot(history['val_loc_loss'], label='Localization', marker='s')
    axes[1, 1].plot(history['val_seg_loss'], label='Segmentation', marker='^')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation: Component Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # F1 Score
    axes[1, 2].plot(history['val_f1'], label='Val F1', marker='o', color='green')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_title('Validation F1 Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(cm, save_path, title='Detection Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Aneurysm', 'Aneurysm'],
                yticklabels=['No Aneurysm', 'Aneurysm'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_location_aucs(location_aucs, save_path, title='AUC per Anatomical Location'):
    """Plot AUC for each location"""
    plt.figure(figsize=(14, 7))
    x = np.arange(len(LOCATION_LABELS))
    bars = plt.bar(x, location_aucs, color='steelblue', alpha=0.8)
    
    # Color bars based on AUC value
    for i, (bar, auc) in enumerate(zip(bars, location_aucs)):
        if auc >= 0.8:
            bar.set_color('green')
        elif auc >= 0.6:
            bar.set_color('steelblue')
        elif auc >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Location', fontsize=11)
    plt.ylabel('AUC', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xticks(x, LOCATION_LABELS, rotation=45, ha='right', fontsize=9)
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, label='Random (0.5)')
    plt.axhline(y=0.8, color='g', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (0.8)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Location AUCs plot saved to: {save_path}")


def train_fusion_model(config, checkpoint_paths, num_epochs=30, batch_size=4, learning_rate=1e-5):
    """Train fusion model"""
    
    print("=" * 80)
    print("TRAINING MULTI-MODAL FUSION MODEL")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.data.output_root, f"fusion_{timestamp}_final_experiment")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load pre-trained models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    modality_models = {}
    for modality, checkpoint_path in checkpoint_paths.items():
        print(f"\nLoading {modality} model from: {checkpoint_path}")
        model = MultiTaskResNet50(num_detection_classes=2, num_location_classes=13, 
                                   input_channels=7, pretrained=False)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Freeze for fusion
        modality_models[modality] = model
        print(f"[OK] {modality} model loaded successfully")
    
    # Create fusion model
    print("\nInitializing fusion model...")
    fusion_model = SimpleFusionModel(modality_models, fusion_strategy='weighted')
    fusion_model = fusion_model.to(device)
    
    # Count trainable parameters (only fusion weights)
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Trainable fusion parameters: {trainable_params:,}")
    
    # Prepare dataset - Use selected dataset instead of full train.csv
    print("\nPreparing dataset...")
    # Get the base directory from config (parent of preprocessed dirs)
    aerux_base = os.path.dirname(config.data.preprocessed_cta_dir).replace('Preprocessed_images_2.5D', '').rstrip(os.sep)
    selected_dataset_path = os.path.join(aerux_base, 'data_splits', 'selected_dataset_2000.csv')
    
    if not os.path.exists(selected_dataset_path):
        print(f"ERROR: Selected dataset not found at {selected_dataset_path}")
        print("Please ensure selected_dataset_2000.csv exists in data_splits folder")
        return None, None
    
    df = pd.read_csv(selected_dataset_path)
    print(f"Loaded selected dataset: {len(df)} samples")
    print(f"  CTA: {len(df[df['Modality'] == 'CTA'])}")
    print(f"  MRA: {len(df[df['Modality'] == 'MRA'])}")
    print(f"  MRI: {len(df[df['Modality'].str.contains('MRI')])}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config.data.random_seed,
                                         stratify=df['Aneurysm Present'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config.data.random_seed,
                                       stratify=temp_df['Aneurysm Present'])
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # Base directories - use paths from config
    base_dirs = {
        'CTA': config.data.preprocessed_cta_dir,
        'MRA': config.data.preprocessed_mra_dir,
        'MRI': config.data.preprocessed_mri_dir
    }
    
    # Verify directories exist
    for modality, dir_path in base_dirs.items():
        if not os.path.exists(dir_path):
            print(f"WARNING: Directory not found: {dir_path}")
    
    # Create datasets - use selected dataset CSV which has all metadata
    # Use segmentation directory if available
    seg_dir = config.data.segmentation_masks_dir if hasattr(config.data, 'segmentation_masks_dir') else None
    if seg_dir and not os.path.exists(seg_dir):
        print(f"Warning: Segmentation directory not found: {seg_dir}")
        seg_dir = None
    
    train_dataset = MultiModalDataset(train_df, base_dirs, selected_dataset_path,
                                     segmentation_dir=seg_dir, augment=True)
    val_dataset = MultiModalDataset(val_df, base_dirs, selected_dataset_path,
                                   segmentation_dir=seg_dir, augment=False)
    test_dataset = MultiModalDataset(test_df, base_dirs, selected_dataset_path,
                                    segmentation_dir=seg_dir, augment=False)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
    
    # Loss and optimizer (only train fusion/calibration weights)
    # Increase localization weight to give more importance to location predictions
    # Lower segmentation weight since masks are not available (will use zero masks)
    seg_weight = 0.5 if seg_dir is None else 2.0
    criterion = MultiTaskLoss(detection_weight=1.0, localization_weight=2.5, segmentation_weight=seg_weight)
    
    print(f"\nLoss weights: Detection=1.0, Localization=2.5, Segmentation={seg_weight}")
    if seg_dir is None:
        print("Note: Segmentation masks not available - using reduced segmentation weight")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, fusion_model.parameters()), 
                          lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    # Training history
    history = {
        'train_loss': [], 'train_det_loss': [], 'train_loc_loss': [], 'train_seg_loss': [],
        'train_weighted_auc': [], 'train_det_auc': [],
        'val_loss': [], 'val_det_loss': [], 'val_loc_loss': [], 'val_seg_loss': [],
        'val_weighted_auc': [], 'val_det_auc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }
    
    best_val_weighted_auc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_det_loss, train_loc_loss, train_seg_loss, train_weighted_auc, train_det_auc = train_epoch(
            fusion_model, train_loader, criterion, optimizer, device, scaler=None
        )
        
        # Validate
        val_metrics = validate(fusion_model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_det_loss'].append(train_det_loss)
        history['train_loc_loss'].append(train_loc_loss)
        history['train_seg_loss'].append(train_seg_loss)
        history['train_weighted_auc'].append(train_weighted_auc)
        history['train_det_auc'].append(train_det_auc)
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_det_loss'].append(val_metrics['det_loss'])
        history['val_loc_loss'].append(val_metrics['loc_loss'])
        history['val_seg_loss'].append(val_metrics['seg_loss'])
        history['val_weighted_auc'].append(val_metrics['weighted_auc'])
        history['val_det_auc'].append(val_metrics['detection_auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f} | Weighted AUC: {train_weighted_auc:.4f} | Det AUC: {train_det_auc:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Weighted AUC: {val_metrics['weighted_auc']:.4f} | Det AUC: {val_metrics['detection_auc']:.4f}")
        print(f"Val   - F1: {val_metrics['f1']:.4f}")
        
        # Show top 3 location AUCs
        top_locs = sorted(zip(LOCATION_LABELS, val_metrics['location_aucs']), key=lambda x: x[1], reverse=True)[:3]
        print(f"Val   - Top Locations: {top_locs[0][0][:30]}... ({top_locs[0][1]:.3f}), {top_locs[1][0][:30]}... ({top_locs[1][1]:.3f}), {top_locs[2][0][:30]}... ({top_locs[2][1]:.3f})")
        
        # Save best model
        if val_metrics['weighted_auc'] > best_val_weighted_auc:
            best_val_weighted_auc = val_metrics['weighted_auc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            checkpoint_path = os.path.join(output_dir, 'best_fusion_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_weighted_auc': best_val_weighted_auc,
                'val_metrics': val_metrics,
                'history': history,
                'fusion_strategy': fusion_model.fusion_strategy,
                'config': {'checkpoint_paths': checkpoint_paths}
            }, checkpoint_path)
            
            print(f"\n[OK] New best model saved! (Weighted AUC: {best_val_weighted_auc:.4f})")
            
            # Save best validation confusion matrix
            cm_path = os.path.join(output_dir, 'best_val_confusion_matrix.png')
            plot_confusion_matrix(val_metrics['confusion_matrix'], cm_path, 
                                title='Validation Confusion Matrix (Best Epoch)')
            
            # Save best validation location AUCs
            loc_path = os.path.join(output_dir, 'best_val_location_aucs.png')
            plot_location_aucs(val_metrics['location_aucs'], loc_path,
                             title='Validation AUC per Anatomical Location (Best Epoch)')
        else:
            patience_counter += 1
            print(f"\nNo improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Test on best model
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(fusion_model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Weighted AUC (Competition Metric): {test_metrics['weighted_auc']:.4f}")
    print(f"  Detection AUC:     {test_metrics['detection_auc']:.4f}")
    print(f"  Detection Acc:     {test_metrics['detection_accuracy']:.4f}")
    print(f"  F1 Score:          {test_metrics['f1']:.4f}")
    print(f"  Precision:         {test_metrics['precision']:.4f}")
    print(f"  Recall:            {test_metrics['recall']:.4f}")
    
    # Generate all visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Training history
    history_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_path)
    
    # 2. Test confusion matrix
    test_cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], test_cm_path,
                         title='Test Set Confusion Matrix')
    
    # 3. Test location AUCs
    test_loc_path = os.path.join(output_dir, 'test_location_aucs.png')
    plot_location_aucs(test_metrics['location_aucs'], test_loc_path,
                      title='Test Set: AUC per Anatomical Location')
    
    # 4. Location AUCs comparison (validation vs test)
    comparison_path = os.path.join(output_dir, 'location_aucs_comparison.png')
    
    # Load best validation metrics
    best_checkpoint = torch.load(checkpoint_path, weights_only=False)
    best_val_metrics = best_checkpoint['val_metrics']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    x = np.arange(len(LOCATION_LABELS))
    width = 0.35
    
    # Validation AUCs
    bars1 = ax1.bar(x - width/2, best_val_metrics['location_aucs'], width, 
                    label='Validation', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, test_metrics['location_aucs'], width,
                    label='Test', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Location', fontsize=11)
    ax1.set_ylabel('AUC', fontsize=11)
    ax1.set_title('Validation vs Test: Location AUCs', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(LOCATION_LABELS, rotation=45, ha='right', fontsize=9)
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Random')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # AUC difference
    auc_diff = np.array(test_metrics['location_aucs']) - np.array(best_val_metrics['location_aucs'])
    colors = ['green' if d >= 0 else 'red' for d in auc_diff]
    ax2.bar(x, auc_diff, color=colors, alpha=0.7)
    ax2.set_xlabel('Location', fontsize=11)
    ax2.set_ylabel('AUC Difference (Test - Val)', fontsize=11)
    ax2.set_title('Performance Change: Test vs Validation', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(LOCATION_LABELS, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Location AUCs comparison saved to: {comparison_path}")
    
    # Save results
    results = {
        'test_weighted_auc': float(test_metrics['weighted_auc']),
        'test_detection_auc': float(test_metrics['detection_auc']),
        'test_accuracy': float(test_metrics['detection_accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_location_aucs': {LOCATION_LABELS[i]: float(test_metrics['location_aucs'][i]) 
                              for i in range(len(LOCATION_LABELS))},
        'best_val_weighted_auc': float(best_val_weighted_auc),
        'best_epoch': best_epoch,
        'total_epochs_trained': epoch + 1
    }
    
    results_path = os.path.join(output_dir, 'test_results_fusion.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - training_history.png")
    print(f"  - test_confusion_matrix.png")
    print(f"  - test_location_aucs.png")
    print(f"  - location_aucs_comparison.png")
    print(f"  - best_val_confusion_matrix.png")
    print(f"  - best_val_location_aucs.png")
    print(f"  - test_results_fusion.json")
    print(f"  - best_fusion_model.pth")
    
    return output_dir, results


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Modal Late Fusion Model')
    parser.add_argument('--cta_model', type=str, 
                       required=True,
                       help='Path to trained CTA model checkpoint')
    parser.add_argument('--mra_model', type=str,
                       required=True,
                       help='Path to trained MRA model checkpoint')
    parser.add_argument('--mri_model', type=str,
                       required=True,
                       help='Path to trained MRI model checkpoint')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (can be higher for fusion)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for calibration layers')
    
    args = parser.parse_args()
    
    config = Config()
    
    checkpoint_paths = {
        'CTA': args.cta_model,
        'MRA': args.mra_model,
        'MRI': args.mri_model
    }
    
    # Verify checkpoints exist
    for modality, path in checkpoint_paths.items():
        if not os.path.exists(path):
            print(f"Error: {modality} checkpoint not found at {path}")
            return
    
    output_dir, results = train_fusion_model(
        config, checkpoint_paths,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n" + "=" * 80)
    print("FUSION TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
