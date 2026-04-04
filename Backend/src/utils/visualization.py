"""
Visualization Utilities
Tools for visualizing model outputs, attention maps, segmentation masks,
ROC curves, and training metrics.

Author: Senior Computer Vision Engineer
Date: 2025-01-08
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(
    train_history: List[Dict],
    val_history: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot training and validation metrics over epochs
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_history) + 1)
    
    # Loss plots
    metrics_to_plot = [
        ('total_loss', 'Total Loss'),
        ('detection_loss', 'Detection Loss'),
        ('segmentation_dice', 'Segmentation Dice'),
        ('detection_accuracy', 'Detection Accuracy'),
        ('detection_auc', 'Detection AUC'),
        ('location_accuracy', 'Location Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        train_vals = [h.get(metric, 0) for h in train_history]
        val_vals = [h.get(metric, 0) for h in val_history]
        
        ax.plot(epochs, train_vals, 'o-', label='Train', linewidth=2)
        ax.plot(epochs, val_vals, 's-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Training history saved to: {save_path}")
    
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" ROC curve saved to: {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.close()


def visualize_segmentation(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    attention_map: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Segmentation Result"
):
    """
    Visualize segmentation results
    
    Args:
        image: Input image (H, W) or (H, W, C)
        pred_mask: Predicted mask (H, W)
        true_mask: Ground truth mask (H, W)
        attention_map: Attention map (H, W), optional
        save_path: Path to save figure
        title: Plot title
    """
    ncols = 4 if attention_map is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    
    # Normalize image for display
    if image.ndim == 3:
        image_display = image[:, :, image.shape[2]//2]  # Middle slice
    else:
        image_display = image
    
    image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min() + 1e-8)
    
    # Original image
    axes[0].imshow(image_display, cmap='gray')
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask overlay
    axes[1].imshow(image_display, cmap='gray')
    axes[1].imshow(true_mask, cmap='Reds', alpha=0.5 * (true_mask > 0))
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Predicted mask overlay
    axes[2].imshow(image_display, cmap='gray')
    axes[2].imshow(pred_mask, cmap='Greens', alpha=0.5 * (pred_mask > 0))
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Attention map (if provided)
    if attention_map is not None:
        axes[3].imshow(image_display, cmap='gray')
        axes[3].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[3].set_title('Attention Map', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Segmentation visualization saved to: {save_path}")
    
    plt.close()


def visualize_attention_maps(
    image: np.ndarray,
    attention_maps: Dict[str, np.ndarray],
    modality_weights: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize attention maps for different modalities
    
    Args:
        image: Input image (H, W)
        attention_maps: Dictionary of attention maps {modality: attention_map}
        modality_weights: Modality weighting scores
        save_path: Path to save figure
    """
    num_modalities = len(attention_maps)
    fig, axes = plt.subplots(1, num_modalities + 1, figsize=(5*(num_modalities+1), 5))
    
    # Normalize image
    image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Original image
    axes[0].imshow(image_display, cmap='gray')
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Attention maps for each modality
    for idx, (modality, attn_map) in enumerate(attention_maps.items(), 1):
        axes[idx].imshow(image_display, cmap='gray')
        axes[idx].imshow(attn_map, cmap='hot', alpha=0.6)
        
        title = f'{modality} Attention'
        if modality_weights is not None:
            title += f'\nWeight: {modality_weights[idx-1]:.3f}'
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Multi-Modal Attention Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Attention maps saved to: {save_path}")
    
    plt.close()


def plot_location_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    location_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot distribution of aneurysm locations
    
    Args:
        predictions: Predicted location indices
        targets: True location indices
        location_names: List of location names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # True distribution
    unique, counts = np.unique(targets, return_counts=True)
    axes[0].bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(unique)))
    axes[0].set_xticklabels([location_names[i] if i < len(location_names) else 'Unknown' for i in unique],
                            rotation=45, ha='right')
    axes[0].set_xlabel('Location', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('True Location Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Predicted distribution
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    axes[1].bar(range(len(unique_pred)), counts_pred, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(unique_pred)))
    axes[1].set_xticklabels([location_names[i] if i < len(location_names) else 'Unknown' for i in unique_pred],
                            rotation=45, ha='right')
    axes[1].set_xlabel('Location', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Predicted Location Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Location distribution saved to: {save_path}")
    
    plt.close()


def create_visualization_report(
    metrics: Dict,
    train_history: List[Dict],
    val_history: List[Dict],
    results: Dict,
    location_labels: List[str],
    output_dir: str
):
    """
    Create comprehensive visualization report
    
    Args:
        metrics: Evaluation metrics
        train_history: Training history
        val_history: Validation history
        results: Evaluation results
        location_labels: List of location names
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Generating Visualization Report")
    print("="*80)
    
    # 1. Training history
    if train_history and val_history:
        plot_training_history(
            train_history, val_history,
            save_path=str(output_path / 'training_history.png')
        )
    
    # 2. ROC curve
    if 'detection' in metrics and 'roc_curve' in metrics['detection']:
        plot_roc_curve(
            np.array(metrics['detection']['roc_curve']['fpr']),
            np.array(metrics['detection']['roc_curve']['tpr']),
            metrics['detection']['auc'],
            save_path=str(output_path / 'roc_curve.png')
        )
    
    # 3. Detection confusion matrix
    if 'detection' in metrics and 'confusion_matrix' in metrics['detection']:
        plot_confusion_matrix(
            np.array(metrics['detection']['confusion_matrix']),
            ['No Aneurysm', 'Aneurysm'],
            title='Detection Confusion Matrix',
            save_path=str(output_path / 'detection_confusion_matrix.png')
        )
    
    # 4. Location confusion matrix
    if 'location' in metrics and 'confusion_matrix' in metrics['location']:
        cm = np.array(metrics['location']['confusion_matrix'])
        unique_labels = np.unique(np.concatenate([
            results['location']['predictions'],
            results['location']['targets']
        ]))
        class_names = [location_labels[i] if i < len(location_labels) else 'Unknown' 
                      for i in unique_labels]
        
        plot_confusion_matrix(
            cm,
            class_names,
            title='Location Classification Confusion Matrix',
            save_path=str(output_path / 'location_confusion_matrix.png')
        )
    
    # 5. Location distribution
    if len(results['location']['predictions']) > 0:
        plot_location_distribution(
            np.array(results['location']['predictions']),
            np.array(results['location']['targets']),
            location_labels,
            save_path=str(output_path / 'location_distribution.png')
        )
    
    print(f"\n Visualization report saved to: {output_path}")
    print("="*80)


# Example usage
if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
