import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Multi-Task Training Script for ResNet50
Trains models for Detection + Localization + Segmentation
Implements 4-fold cross-validation
Implements competition metric: Mean Weighted Columnwise AUCROC

Formula: Final Score = 1/2 * (AUC_AP + 1/13 * Σ(AUC_Ci))
where:
- AUC_AP = AUC for Aneurysm Present (detection)
- AUC_Ci = AUC for each of 13 location classes


Usage:
    python train_multitask.py --modality CTA
    python train_multitask.py --modality MRA
    python train_multitask.py --modality MRI
    python train_multitask.py --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import Config

# 13 Location labels - MUST MATCH EXACT COLUMN NAMES IN train.csv
LOCATION_LABELS = [
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


class MultiTaskDataset(Dataset):
    """Dataset for multi-task aneurysm detection, localization, and segmentation"""
    
    def __init__(self, dataframe, modality, base_dir, train_csv_path, segmentation_dir=None, augment=False):
        """
        Args:
            dataframe: DataFrame with SeriesInstanceUID
            modality: 'CTA', 'MRA', or 'MRI'
            base_dir: Base directory containing preprocessed .npy files
            train_csv_path: Path to train.csv with location labels
            segmentation_dir: Directory with segmentation masks (optional)
            augment: Apply data augmentation
        """
        self.df = dataframe
        self.modality = modality
        self.base_dir = base_dir
        self.segmentation_dir = segmentation_dir
        self.augment = augment
        
        # Load full train.csv for location labels
        self.train_csv = pd.read_csv(train_csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def _get_location_labels(self, series_uid):
        """Get multi-label location vector for a series"""
        # Find row for this series (each SeriesInstanceUID appears once)
        series_row = self.train_csv[self.train_csv['SeriesInstanceUID'] == series_uid]
        
        # Create 13-dimensional binary vector
        location_vector = np.zeros(13, dtype=np.float32)
        
        if len(series_row) == 0:
            return location_vector  # No data found
        
        # Get the first (and only) row
        row = series_row.iloc[0]
        
        # Check if aneurysm is present
        if pd.isna(row.get('Aneurysm Present', 0)) or row.get('Aneurysm Present', 0) == 0:
            return location_vector  # No aneurysm, all zeros
        
        # Mark present locations using exact column names
        for i, label in enumerate(LOCATION_LABELS):
            # Use exact column name from CSV
            if label in row and pd.notna(row[label]) and row[label] == 1:
                location_vector[i] = 1.0
        
        return location_vector
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        detection_label = int(row['Aneurysm Present'])
        
        # Construct file path
        label_name = 'aneurysm' if detection_label == 1 else 'no_aneurysm'
        file_path = os.path.join(self.base_dir, label_name, f"{series_uid}.npy")
        
        # Load preprocessed volume
        try:
            volume = np.load(file_path).astype(np.float32)  # Shape: (256, 256, 7)
            
            # Transpose to (7, 256, 256) for PyTorch
            volume = np.transpose(volume, (2, 0, 1))
            
            # Get location labels
            location_labels = self._get_location_labels(series_uid)
            
            # Load segmentation mask if available
            if self.segmentation_dir and detection_label == 1:
                mask_path = os.path.join(self.segmentation_dir, f"{series_uid}.npy")
                if os.path.exists(mask_path):
                    mask = np.load(mask_path).astype(np.float32)
                    # Ensure mask is (256, 256)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]  # Take first channel if multi-channel
                else:
                    mask = np.zeros((256, 256), dtype=np.float32)
            else:
                mask = np.zeros((256, 256), dtype=np.float32)
            
            # Apply augmentation if enabled
            if self.augment:
                volume, mask = self._augment(volume, mask)
            
            return (
                torch.from_numpy(volume),
                torch.tensor(detection_label, dtype=torch.long),
                torch.from_numpy(location_labels),
                torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            )
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero volume if file not found
            return (
                torch.zeros((7, 256, 256), dtype=torch.float32),
                torch.tensor(detection_label, dtype=torch.long),
                torch.zeros(13, dtype=torch.float32),
                torch.zeros((1, 256, 256), dtype=torch.float32)
            )
    
    def _augment(self, volume, mask):
        """Simple data augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            volume = np.rot90(volume, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
        
        return volume, mask


class MultiTaskResNet50(nn.Module):
    """ResNet50 with multiple task heads: Detection, Localization, and Segmentation"""
    
    def __init__(self, num_detection_classes=2, num_location_classes=13, input_channels=7, pretrained=True):
        super(MultiTaskResNet50, self).__init__()
        
        # Load pretrained ResNet50
        from torchvision import models
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)
        
        # Modify first conv layer to accept 7 channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new conv1 weights
        if pretrained:
            pretrained_weights = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weights.repeat(1, input_channels // 3 + 1, 1, 1)[:, :input_channels, :, :]
        
        # Encoder layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        self.avgpool = resnet.avgpool
        
        # Detection head (binary classification)
        self.detection_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_detection_classes)
        )
        
        # Localization head (multi-label classification for 13 locations)
        self.localization_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_location_classes)
        )
        
        # Segmentation decoder (U-Net style)
        # Input spatial size: 256x256
        # After conv1 (stride=2): 128x128
        # After maxpool (stride=2): 64x64 (x0)
        # After layer1: 64x64 (x1)
        # After layer2 (stride=2): 32x32 (x2)
        # After layer3 (stride=2): 16x16 (x3)
        # After layer4 (stride=2): 8x8 (x4)
        
        self.seg_upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.seg_conv4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.seg_upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.seg_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.seg_upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.seg_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.seg_upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)  # 64x64 -> 128x128
        self.seg_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 128x128 -> 256x256
            nn.Conv2d(32, 1, kernel_size=1)
            # No sigmoid here - will be applied in loss function for numerical stability
        )
    
    def forward(self, x):
        # Encoder with proper feature map saving
        # Input: 256x256
        x_conv = self.conv1(x)  # 128x128
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)
        x0 = self.maxpool(x_conv)  # 64x64
        
        x1 = self.layer1(x0)  # 64x64 (no downsampling in layer1)
        x2 = self.layer2(x1)  # 32x32
        x3 = self.layer3(x2)  # 16x16
        x4 = self.layer4(x3)  # 8x8
        
        # Detection and Localization (use global pooled features)
        pooled = self.avgpool(x4)
        pooled = torch.flatten(pooled, 1)
        
        detection_logits = self.detection_head(pooled)
        localization_logits = self.localization_head(pooled)
        
        # Segmentation decoder with proper skip connections
        seg = self.seg_upconv4(x4)  # 8x8 -> 16x16
        seg = torch.cat([seg, x3], dim=1)  # concat with x3 (16x16)
        seg = self.seg_conv4(seg)
        
        seg = self.seg_upconv3(seg)  # 16x16 -> 32x32
        seg = torch.cat([seg, x2], dim=1)  # concat with x2 (32x32)
        seg = self.seg_conv3(seg)
        
        seg = self.seg_upconv2(seg)  # 32x32 -> 64x64
        seg = torch.cat([seg, x1], dim=1)  # concat with x1 (64x64)
        seg = self.seg_conv2(seg)
        
        seg = self.seg_upconv1(seg)  # 64x64 -> 128x128
        seg = torch.cat([seg, x_conv], dim=1)  # concat with x_conv (128x128) not x0!
        seg = self.seg_conv1(seg)
        
        segmentation_mask = self.seg_final(seg)  # 128x128 -> 256x256
        
        return detection_logits, localization_logits, segmentation_mask


class MultiTaskLoss(nn.Module):
    """Combined loss for detection, localization, and segmentation"""
    
    def __init__(self, detection_weight=1.0, localization_weight=1.0, segmentation_weight=2.0):
        super(MultiTaskLoss, self).__init__()
        self.detection_weight = detection_weight
        self.localization_weight = localization_weight
        self.segmentation_weight = segmentation_weight
        
        self.detection_criterion = nn.CrossEntropyLoss()
        self.localization_criterion = nn.BCEWithLogitsLoss()  # Multi-label
        self.segmentation_criterion = self._dice_bce_loss
    
    def _dice_bce_loss(self, pred_logits, target):
        """Dice + BCE combined loss for segmentation"""
        # Use BCEWithLogitsLoss for numerical stability with autocast
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')
        
        # Dice loss (apply sigmoid to logits first)
        smooth = 1e-5
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return bce + dice
    
    def forward(self, detection_pred, localization_pred, segmentation_pred, 
                detection_target, localization_target, segmentation_target):
        
        detection_loss = self.detection_criterion(detection_pred, detection_target)
        localization_loss = self.localization_criterion(localization_pred, localization_target)
        segmentation_loss = self.segmentation_criterion(segmentation_pred, segmentation_target)
        
        total_loss = (self.detection_weight * detection_loss + 
                     self.localization_weight * localization_loss + 
                     self.segmentation_weight * segmentation_loss)
        
        return total_loss, detection_loss, localization_loss, segmentation_loss


def compute_weighted_auc(detection_probs, detection_labels, location_probs, location_labels):
    """
    Compute Mean Weighted Columnwise AUCROC
    Final Score = 1/2 * (AUC_AP + 1/13 * Σ(AUC_Ci))
    """
    # Detection AUC (Aneurysm Present)
    try:
        auc_ap = roc_auc_score(detection_labels, detection_probs)
    except:
        auc_ap = 0.5
    
    # Location AUCs (for each of 13 classes)
    location_aucs = []
    for i in range(13):
        try:
            if len(np.unique(location_labels[:, i])) > 1:  # Check if both classes present
                auc_i = roc_auc_score(location_labels[:, i], location_probs[:, i])
                location_aucs.append(auc_i)
            else:
                location_aucs.append(0.5)  # Default if only one class
        except:
            location_aucs.append(0.5)
    
    # Weighted average
    avg_location_auc = np.mean(location_aucs)
    final_score = 0.5 * (auc_ap + (1.0 / 13.0) * sum(location_aucs))
    
    return final_score, auc_ap, location_aucs


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
    
    for batch_idx, (inputs, det_labels, loc_labels, seg_masks) in enumerate(progress_bar):
        inputs = inputs.to(device)
        det_labels = det_labels.to(device)
        loc_labels = loc_labels.to(device)
        seg_masks = seg_masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast('cuda'):
                det_logits, loc_logits, seg_pred = model(inputs)
                total_loss, det_loss, loc_loss, seg_loss = criterion(
                    det_logits, loc_logits, seg_pred,
                    det_labels, loc_labels, seg_masks
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            det_logits, loc_logits, seg_pred = model(inputs)
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
        
        # Update progress bar
        progress_bar.set_postfix({'loss': total_loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    det_loss = running_det_loss / len(dataloader)
    loc_loss = running_loc_loss / len(dataloader)
    seg_loss = running_seg_loss / len(dataloader)
    
    # Compute metrics
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
        for inputs, det_labels, loc_labels, seg_masks in tqdm(dataloader, desc='Validating'):
            inputs = inputs.to(device)
            det_labels = det_labels.to(device)
            loc_labels = loc_labels.to(device)
            seg_masks = seg_masks.to(device)
            
            det_logits, loc_logits, seg_pred = model(inputs)
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
    
    # Compute metrics
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
    
    metrics = {
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
    
    return metrics


def plot_training_history(history, save_path):
    """Plot training history"""
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


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Aneurysm', 'Aneurysm'],
                yticklabels=['No Aneurysm', 'Aneurysm'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Detection Confusion Matrix (Best Validation Epoch)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_location_aucs(location_aucs, save_path):
    """Plot AUC for each location"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(LOCATION_LABELS))
    plt.bar(x, location_aucs, color='steelblue', alpha=0.8)
    plt.xlabel('Location')
    plt.ylabel('AUC')
    plt.title('AUC per Anatomical Location')
    plt.xticks(x, LOCATION_LABELS, rotation=45, ha='right')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Location AUCs plot saved to: {save_path}")


def train_single_modality(modality, config, num_epochs=50, batch_size=8, learning_rate=1e-4, n_folds=4):
    """Train a single modality multi-task model with k-fold cross-validation"""
    
    print("=" * 80)
    print(f"TRAINING MULTI-TASK RESNET50 MODEL FOR {modality} WITH {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.data.output_root, "multitask_kfold_new",f"multitask_{modality}_{timestamp}_kfold")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base directory for modality
    if modality == 'CTA':
        base_dir = config.data.preprocessed_cta_dir
    elif modality == 'MRA':
        base_dir = config.data.preprocessed_mra_dir
    else:  # MRI
        base_dir = config.data.preprocessed_mri_dir
    
    print(f"\nData directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data splits (global train only when use_global_split — see create_global_split.py)
    print("\nLoading data splits...")
    from src.datasets.global_split_utils import load_multitask_source_dataframe

    df = load_multitask_source_dataframe(config)
    
    # Filter by modality
    if modality == 'MRI':
        df_modality = df[df['Modality'].str.contains('MRI', case=False, na=False)].copy()
    else:
        df_modality = df[df['Modality'].str.upper() == modality].copy()
    
    print(f"Filtered {modality} data: {len(df_modality)} series")
    
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.data.random_seed)
    
    # Store results for all folds
    fold_results = []
    all_fold_histories = []
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Start k-fold cross-validation
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(df_modality)):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print("=" * 80)
        
        # Create fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data for this fold
        train_df = df_modality.iloc[train_indices].copy()
        val_df = df_modality.iloc[val_indices].copy()
        
        print(f"\nFold {fold_idx + 1} data splits:")
        print(f"  Train: {len(train_df)} ({train_df['Aneurysm Present'].sum()} positive)")
        print(f"  Val:   {len(val_df)} ({val_df['Aneurysm Present'].sum()} positive)")
        
        # Create datasets for this fold
        train_dataset = MultiTaskDataset(
            train_df, modality, base_dir, config.data.train_csv,
            segmentation_dir=config.data.segmentation_masks_dir, augment=True
        )
        val_dataset = MultiTaskDataset(
            val_df, modality, base_dir, config.data.train_csv,
            segmentation_dir=config.data.segmentation_masks_dir, augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)
        
        # Initialize model for this fold
        model = MultiTaskResNet50(num_detection_classes=2, num_location_classes=13, 
                                   input_channels=7, pretrained=True)
        model = model.to(device)
        
        if fold_idx == 0:  # Print model info only for first fold
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nModel parameters:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
        
        # Loss and optimizer
        criterion = MultiTaskLoss(detection_weight=1.0, localization_weight=1.0, segmentation_weight=2.0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
        
        # Mixed precision scaler
        scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Training history for this fold
        history = {
            'train_loss': [], 'train_det_loss': [], 'train_loc_loss': [], 'train_seg_loss': [],
            'train_weighted_auc': [], 'train_det_auc': [],
            'val_loss': [], 'val_det_loss': [], 'val_loc_loss': [], 'val_seg_loss': [],
            'val_weighted_auc': [], 'val_det_auc': [], 'val_f1': [],
            'val_precision': [], 'val_recall': []
        }
        
        best_val_weighted_auc = 0.0
        best_epoch = 0
        patience = 15
        patience_counter = 0
        
        # Training loop for this fold
        print(f"\nStarting training for {num_epochs} epochs...")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            print(f"\nFold {fold_idx + 1}/{n_folds} - Epoch {epoch+1}/{num_epochs}")
            print("-" * 40)
        
            # Train
            train_loss, train_det_loss, train_loc_loss, train_seg_loss, train_weighted_auc, train_det_auc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
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
            print(f"Val   - F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            
            # Save best model for this fold
            if val_metrics['weighted_auc'] > best_val_weighted_auc:
                best_val_weighted_auc = val_metrics['weighted_auc']
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save model
                checkpoint_path = os.path.join(fold_dir, f'best_model_{modality}_fold{fold_idx + 1}.pth')
                torch.save({
                    'fold': fold_idx + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_weighted_auc': best_val_weighted_auc,
                    'val_metrics': val_metrics,
                    'history': history
                }, checkpoint_path)
                
                print(f"\n✓ New best model saved! (Weighted AUC: {best_val_weighted_auc:.4f})")
                
                # Save confusion matrix and location AUCs for best epoch
                best_cm = val_metrics['confusion_matrix']
                best_loc_aucs = val_metrics['location_aucs']
            else:
                patience_counter += 1
                print(f"\nNo improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save fold training history
        print("\n" + "-" * 80)
        print(f"FOLD {fold_idx + 1} TRAINING COMPLETED")
        print("-" * 80)
        print(f"\nBest Validation Weighted AUC: {best_val_weighted_auc:.4f} (Epoch {best_epoch})")
        
        # Save training plots for this fold
        plot_path = os.path.join(fold_dir, f'training_history_fold{fold_idx + 1}.png')
        plot_training_history(history, plot_path)
        
        cm_path = os.path.join(fold_dir, f'confusion_matrix_fold{fold_idx + 1}.png')
        plot_confusion_matrix(best_cm, cm_path)
        
        loc_auc_path = os.path.join(fold_dir, f'location_aucs_fold{fold_idx + 1}.png')
        plot_location_aucs(best_loc_aucs, loc_auc_path)
        
        # Save history as JSON
        history_path = os.path.join(fold_dir, f'history_fold{fold_idx + 1}.json')
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Evaluate on validation set with best model
        print("\n" + "-" * 80)
        print(f"EVALUATING FOLD {fold_idx + 1} ON VALIDATION SET")
        print("-" * 80)
        
        # Load best model for this fold
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        val_final_metrics = validate(model, val_loader, criterion, device)
        
        print(f"\nValidation Results (Fold {fold_idx + 1}):")
        print(f"  Weighted AUC (Competition Metric): {val_final_metrics['weighted_auc']:.4f}")
        print(f"  Detection AUC:     {val_final_metrics['detection_auc']:.4f}")
        print(f"  Detection Acc:     {val_final_metrics['detection_accuracy']:.4f}")
        print(f"  Precision:         {val_final_metrics['precision']:.4f}")
        print(f"  Recall:            {val_final_metrics['recall']:.4f}")
        print(f"  F1 Score:          {val_final_metrics['f1']:.4f}")
        
        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'modality': modality,
            'val_weighted_auc': float(val_final_metrics['weighted_auc']),
            'val_detection_auc': float(val_final_metrics['detection_auc']),
            'val_accuracy': float(val_final_metrics['detection_accuracy']),
            'val_precision': float(val_final_metrics['precision']),
            'val_recall': float(val_final_metrics['recall']),
            'val_f1': float(val_final_metrics['f1']),
            'val_location_aucs': [float(x) for x in val_final_metrics['location_aucs']],
            'best_val_weighted_auc': float(best_val_weighted_auc),
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1
        }
        
        fold_results.append(fold_result)
        all_fold_histories.append(history)
        
        results_path = os.path.join(fold_dir, f'fold{fold_idx + 1}_results.json')
        with open(results_path, 'w') as f:
            json.dump(fold_result, f, indent=2)
        print(f"\nFold {fold_idx + 1} results saved to: {results_path}")
    
    # Calculate average metrics across all folds
    print("\n" + "=" * 80)
    print(f"{n_folds}-FOLD CROSS-VALIDATION COMPLETED")
    print("=" * 80)
    
    avg_metrics = {
        'modality': modality,
        'n_folds': n_folds,
        'avg_weighted_auc': np.mean([r['val_weighted_auc'] for r in fold_results]),
        'std_weighted_auc': np.std([r['val_weighted_auc'] for r in fold_results]),
        'avg_detection_auc': np.mean([r['val_detection_auc'] for r in fold_results]),
        'std_detection_auc': np.std([r['val_detection_auc'] for r in fold_results]),
        'avg_accuracy': np.mean([r['val_accuracy'] for r in fold_results]),
        'std_accuracy': np.std([r['val_accuracy'] for r in fold_results]),
        'avg_precision': np.mean([r['val_precision'] for r in fold_results]),
        'std_precision': np.std([r['val_precision'] for r in fold_results]),
        'avg_recall': np.mean([r['val_recall'] for r in fold_results]),
        'std_recall': np.std([r['val_recall'] for r in fold_results]),
        'avg_f1': np.mean([r['val_f1'] for r in fold_results]),
        'std_f1': np.std([r['val_f1'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    print(f"\nCross-Validation Results ({n_folds} folds):")
    print(f"  Weighted AUC (Competition):  {avg_metrics['avg_weighted_auc']:.4f} ± {avg_metrics['std_weighted_auc']:.4f}")
    print(f"  Detection AUC:               {avg_metrics['avg_detection_auc']:.4f} ± {avg_metrics['std_detection_auc']:.4f}")
    print(f"  Accuracy:                    {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"  Precision:                   {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"  Recall:                      {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    print(f"  F1 Score:                    {avg_metrics['avg_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    
    print(f"\nIndividual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Weighted AUC = {result['val_weighted_auc']:.4f}, F1 = {result['val_f1']:.4f}")
    
    # Save overall cross-validation results
    cv_results_path = os.path.join(output_dir, f'cross_validation_results_{modality}.json')
    with open(cv_results_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nCross-validation summary saved to: {cv_results_path}")
    
    print("\n" + "=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)
    
    return output_dir, avg_metrics


def validate_cv_completed(cv_dir, modality, n_folds=4):
    """Validate that K-fold cross-validation has been completed.
    
    Checks that:
    - cv_dir exists
    - cross_validation_results_{modality}.json exists
    - Each fold's best_model checkpoint exists
    
    Returns the parsed CV results dict on success, or exits with an error.
    """
    if not os.path.isdir(cv_dir):
        print(f"\n✗ ERROR: Cross-validation directory not found: {cv_dir}")
        print(f"\n  You must complete K-fold CV first by running:")
        print(f"    python {os.path.basename(__file__)} --modality {modality}")
        print(f"\n  Then use the output directory as --cv_dir.")
        sys.exit(1)
    
    cv_results_path = os.path.join(cv_dir, f'cross_validation_results_{modality}.json')
    if not os.path.isfile(cv_results_path):
        print(f"\n✗ ERROR: CV results file not found: {cv_results_path}")
        print(f"  The K-fold cross-validation does not appear to be completed for {modality}.")
        print(f"\n  Run K-fold CV first:")
        print(f"    python {os.path.basename(__file__)} --modality {modality}")
        sys.exit(1)
    
    with open(cv_results_path, 'r') as f:
        cv_results = json.load(f)
    
    # Check each fold's best model
    missing_models = []
    for fold_idx in range(1, n_folds + 1):
        fold_dir = os.path.join(cv_dir, f'fold_{fold_idx}')
        model_path = os.path.join(fold_dir, f'best_model_{modality}_fold{fold_idx}.pth')
        if not os.path.isfile(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print(f"\n✗ ERROR: The following fold model checkpoints are missing:")
        for p in missing_models:
            print(f"    {p}")
        print(f"\n  K-fold cross-validation is incomplete for {modality}.")
        print(f"  Run it first:")
        print(f"    python {os.path.basename(__file__)} --modality {modality}")
        sys.exit(1)
    
    print(f"\n✓ Cross-validation results validated successfully.")
    print(f"  CV directory: {cv_dir}")
    print(f"  All {n_folds} fold models found.")
    
    return cv_results


def train_final_model(modality, config, cv_dir, batch_size=8, learning_rate=1e-4, n_folds=4):
    """Train a final model on the entire training pool and evaluate on held-out test set.
    
    Uses the median best epoch from CV as the fixed number of training epochs.
    No early stopping — the epoch count is determined from CV results.
    """
    
    # Validate CV is completed
    cv_results = validate_cv_completed(cv_dir, modality, n_folds)
    
    # Extract median best epoch from CV folds
    best_epochs = [r['best_epoch'] for r in cv_results['fold_results']]
    median_epochs = int(np.median(best_epochs))
    
    print("\n" + "=" * 80)
    print(f"TRAINING FINAL RESNET50 MODEL FOR {modality}")
    print("=" * 80)
    print(f"\nCV fold best epochs: {best_epochs}")
    print(f"Median best epoch (used for final training): {median_epochs}")
    print(f"\nCV Performance Summary:")
    print(f"  Weighted AUC: {cv_results['avg_weighted_auc']:.4f} ± {cv_results['std_weighted_auc']:.4f}")
    print(f"  Detection AUC: {cv_results['avg_detection_auc']:.4f} ± {cv_results['std_detection_auc']:.4f}")
    print(f"  F1 Score: {cv_results['avg_f1']:.4f} ± {cv_results['std_f1']:.4f}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.data.output_root, f"multitask_{modality}_{timestamp}_final")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base directory for modality
    if modality == 'CTA':
        base_dir = config.data.preprocessed_cta_dir
    elif modality == 'MRA':
        base_dir = config.data.preprocessed_mra_dir
    else:  # MRI
        base_dir = config.data.preprocessed_mri_dir
    
    print(f"\nData directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load FULL training pool (all data allowed for training)
    print("\nLoading full training pool...")
    from src.datasets.global_split_utils import load_multitask_source_dataframe
    df = load_multitask_source_dataframe(config)
    
    # Filter by modality
    if modality == 'MRI':
        df_modality = df[df['Modality'].str.contains('MRI', case=False, na=False)].copy()
    else:
        df_modality = df[df['Modality'].str.upper() == modality].copy()
    
    print(f"Full training pool for {modality}: {len(df_modality)} series")
    print(f"  Positive (aneurysm): {int(df_modality['Aneurysm Present'].sum())}")
    print(f"  Negative (no aneurysm): {int(len(df_modality) - df_modality['Aneurysm Present'].sum())}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataset on ALL training data
    train_dataset = MultiTaskDataset(
        df_modality, modality, base_dir, config.data.train_csv,
        segmentation_dir=config.data.segmentation_masks_dir, augment=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    
    # Initialize model
    model = MultiTaskResNet50(num_detection_classes=2, num_location_classes=13,
                               input_channels=7, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss(detection_weight=1.0, localization_weight=1.0, segmentation_weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=median_epochs, eta_min=1e-7)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Training history
    history = {
        'train_loss': [], 'train_det_loss': [], 'train_loc_loss': [], 'train_seg_loss': [],
        'train_weighted_auc': [], 'train_det_auc': []
    }
    
    # Training loop — fixed epoch count, no early stopping
    print(f"\nStarting FINAL training for {median_epochs} epochs (median best epoch from CV)...")
    print("NOTE: No validation-based early stopping — epoch count fixed from CV.")
    print("-" * 80)
    
    for epoch in range(median_epochs):
        print(f"\nEpoch {epoch+1}/{median_epochs}")
        print("-" * 40)
        
        train_loss, train_det_loss, train_loc_loss, train_seg_loss, train_weighted_auc, train_det_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_det_loss'].append(train_det_loss)
        history['train_loc_loss'].append(train_loc_loss)
        history['train_seg_loss'].append(train_seg_loss)
        history['train_weighted_auc'].append(train_weighted_auc)
        history['train_det_auc'].append(train_det_auc)
        
        print(f"Train - Loss: {train_loss:.4f} | Weighted AUC: {train_weighted_auc:.4f} | Det AUC: {train_det_auc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'final_model_{modality}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_epochs': median_epochs,
        'cv_best_epochs': best_epochs,
        'cv_dir': cv_dir,
        'modality': modality,
        'history': history
    }, final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f'final_training_history_{modality}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training loss curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], marker='o', color='steelblue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Final Training: Total Loss')
    axes[0].grid(True)
    
    axes[1].plot(history['train_weighted_auc'], marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Weighted AUC')
    axes[1].set_title('Final Training: Weighted AUC')
    axes[1].grid(True)
    
    axes[2].plot(history['train_det_loss'], label='Detection', marker='o')
    axes[2].plot(history['train_loc_loss'], label='Localization', marker='s')
    axes[2].plot(history['train_seg_loss'], label='Segmentation', marker='^')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Final Training: Component Losses')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'final_training_history_{modality}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== EVALUATE ON HELD-OUT TEST SET ==========
    print("\n" + "=" * 80)
    print(f"EVALUATING FINAL MODEL ON HELD-OUT TEST SET")
    print("=" * 80)
    
    from src.datasets.global_split_utils import load_global_test_dataframe
    
    if not getattr(config.data, 'use_global_split', False):
        print("\n⚠ WARNING: use_global_split is False. No held-out test set available.")
        print("  Skipping test evaluation. Set use_global_split=True and run create_global_split.py first.")
        print(f"\nAll outputs saved to: {output_dir}")
        return output_dir
    
    test_df = load_global_test_dataframe(config)
    
    # Filter test set by modality
    from preprocess_dataset import normalize_modality
    test_df = test_df.copy()
    test_df['Modality'] = test_df['Modality'].apply(normalize_modality)
    
    if modality == 'MRI':
        test_df_modality = test_df[test_df['Modality'].str.contains('MRI', case=False, na=False)].copy()
    else:
        test_df_modality = test_df[test_df['Modality'].str.upper() == modality].copy()
    
    print(f"\nTest set for {modality}: {len(test_df_modality)} series")
    print(f"  Positive (aneurysm): {int(test_df_modality['Aneurysm Present'].sum())}")
    print(f"  Negative (no aneurysm): {int(len(test_df_modality) - test_df_modality['Aneurysm Present'].sum())}")
    
    if len(test_df_modality) == 0:
        print(f"\n⚠ WARNING: No test samples found for {modality}. Skipping test evaluation.")
        print(f"\nAll outputs saved to: {output_dir}")
        return output_dir
    
    # Create test dataset and loader
    test_dataset = MultiTaskDataset(
        test_df_modality, modality, base_dir, config.data.train_csv,
        segmentation_dir=config.data.segmentation_masks_dir, augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # Run evaluation
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"HELD-OUT TEST SET RESULTS — {modality} (Final Model)")
    print(f"{'='*60}")
    print(f"  Weighted AUC (Competition): {test_metrics['weighted_auc']:.4f}")
    print(f"  Detection AUC:              {test_metrics['detection_auc']:.4f}")
    print(f"  Detection Accuracy:         {test_metrics['detection_accuracy']:.4f}")
    print(f"  Precision:                  {test_metrics['precision']:.4f}")
    print(f"  Recall:                     {test_metrics['recall']:.4f}")
    print(f"  F1 Score:                   {test_metrics['f1']:.4f}")
    print(f"{'='*60}")
    
    # Save test metrics
    test_results = {
        'modality': modality,
        'evaluation_split': 'global_held_out_test',
        'training_epochs': median_epochs,
        'cv_best_epochs': best_epochs,
        'cv_dir': cv_dir,
        'cv_avg_weighted_auc': cv_results['avg_weighted_auc'],
        'cv_std_weighted_auc': cv_results['std_weighted_auc'],
        'test_samples': len(test_df_modality),
        'test_weighted_auc': float(test_metrics['weighted_auc']),
        'test_detection_auc': float(test_metrics['detection_auc']),
        'test_accuracy': float(test_metrics['detection_accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_location_aucs': [float(x) for x in test_metrics['location_aucs']],
        'test_loss': float(test_metrics['loss']),
    }
    
    test_results_path = os.path.join(output_dir, f'final_test_results_{modality}.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {test_results_path}")
    
    # Plot confusion matrix and location AUCs for test set
    cm_path = os.path.join(output_dir, f'test_confusion_matrix_{modality}.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], cm_path)
    
    loc_auc_path = os.path.join(output_dir, f'test_location_aucs_{modality}.png')
    plot_location_aucs(test_metrics['location_aucs'], loc_auc_path)
    
    print(f"\n" + "=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print(f"  Final model:     {final_model_path}")
    print(f"  Test results:    {test_results_path}")
    print("=" * 80)
    
    return output_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Multi-Task ResNet50 for Aneurysm Detection/Localization/Segmentation')
    parser.add_argument('--modality', type=str, choices=['CTA', 'MRA', 'MRI'], 
                       help='Modality to train (CTA, MRA, or MRI)')
    parser.add_argument('--all', action='store_true', 
                       help='Train all three modalities sequentially')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size (default: 8, reduced for multi-task)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--train_final', action='store_true',
                       help='Train a FINAL model on all training data (requires completed K-fold CV via --cv_dir)')
    parser.add_argument('--cv_dir', type=str, default=None,
                       help='Path to completed K-fold CV output directory (required with --train_final)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Determine which modalities to train
    if args.all:
        modalities = ['CTA', 'MRA', 'MRI']
    elif args.modality:
        modalities = [args.modality]
    else:
        print("Error: Please specify --modality or --all")
        return
    
    # ===== FINAL MODEL MODE =====
    if args.train_final:
        if args.cv_dir is None:
            print("\n✗ ERROR: --cv_dir is required when using --train_final.")
            print("  Provide the path to a completed K-fold CV output directory.")
            print(f"\n  Example:")
            print(f"    python {os.path.basename(__file__)} --modality CTA --train_final --cv_dir outputs/multitask_CTA_..._kfold")
            sys.exit(1)
        
        for modality in modalities:
            train_final_model(
                modality, config, args.cv_dir,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        return
    
    # ===== K-FOLD CROSS-VALIDATION MODE (default) =====
    all_results = {}
    for modality in modalities:
        output_dir, results = train_single_modality(
            modality, config, 
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        all_results[modality] = results
        print("\n" + "=" * 80)
        print("\n")
    
    # Summary if training all modalities
    if len(modalities) > 1:
        print("=" * 80)
        print("SUMMARY OF ALL MODALITIES (Cross-Validation Results)")
        print("=" * 80)
        for modality, results in all_results.items():
            print(f"\n{modality}:")
            print(f"  Weighted AUC (Competition): {results['avg_weighted_auc']:.4f} ± {results['std_weighted_auc']:.4f}")
            print(f"  Detection AUC:              {results['avg_detection_auc']:.4f} ± {results['std_detection_auc']:.4f}")
            print(f"  F1 Score:                   {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")
    
    print("\n" + "-" * 80)
    print("NEXT STEP: Train final model on all training data")
    print("-" * 80)
    print(f"\nTo train a final model using these CV results, run:")
    for modality in modalities:
        print(f"  python {os.path.basename(__file__)} --modality {modality} --train_final --cv_dir {output_dir}")
    print()


if __name__ == "__main__":
    main()
