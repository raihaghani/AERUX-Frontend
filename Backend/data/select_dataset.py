"""
Dataset Selection Script for Intracranial Aneurysm Detection

This script selects a balanced subset of 2000 series from the full dataset:
- 1000 CTA: 500 with aneurysm, 500 without
- 500 MRA: 250 with aneurysm, 250 without  
- 500 MRI: 250 with aneurysm (125 T1post + 125 T2), 250 without (125 T1post + 125 T2)

Also filters out erroneous series listed in error_data.yaml

Author: Aerux Final Project
Date: December 2025
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, Dict, List, Tuple
import logging
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_error_series_ids(yaml_path: str) -> Set[str]:
    """
    Load erroneous SeriesInstanceUIDs from error_data.yaml.
    
    Args:
        yaml_path: Path to error_data.yaml file
        
    Returns:
        Set of error SeriesInstanceUIDs to exclude
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert to set if it's a list
        if isinstance(data, list):
            series_ids = set(data)
        else:
            series_ids = set()
            logger.warning("YAML file format is not as expected")
        
        logger.info(f"Loaded {len(series_ids)} error series IDs to exclude")
        return series_ids
        
    except FileNotFoundError:
        logger.warning(f"Error data file not found: {yaml_path}")
        return set()
    except Exception as e:
        logger.error(f"Error loading error_data.yaml: {e}")
        return set()


def normalize_modality(modality: str) -> str:
    """
    Normalize modality names to standard format.
    Combines MRI T1post and MRI T2 into 'MRI' category for processing.
    
    Args:
        modality: Original modality string
        
    Returns:
        Normalized modality name
    """
    modality_upper = str(modality).upper().strip()
    
    if 'CTA' in modality_upper:
        return 'CTA'
    elif 'MRA' in modality_upper:
        return 'MRA'
    elif 'T1' in modality_upper or 'T1POST' in modality_upper:
        return 'MRI_T1post'
    elif 'T2' in modality_upper:
        return 'MRI_T2'
    elif 'MRI' in modality_upper:
        return 'MRI_T1post'  # Default to T1 if not specified
    else:
        return 'Unknown'


def select_balanced_dataset(
    train_csv_path: str,
    error_yaml_path: str = None,
    output_csv_path: str = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Select a balanced subset of 2000 series from the full dataset.
    
    Target Distribution:
    - 1000 CTA: 500 aneurysm, 500 no aneurysm
    - 500 MRA: 250 aneurysm, 250 no aneurysm
    - 500 MRI: 125 T1 aneurysm, 125 T1 no aneurysm, 125 T2 aneurysm, 125 T2 no aneurysm
    
    Args:
        train_csv_path: Path to train.csv
        error_yaml_path: Path to error_data.yaml (optional)
        output_csv_path: Path to save selected dataset CSV (optional)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with selected series
    """
    logger.info("=" * 80)
    logger.info("DATASET SELECTION STARTED")
    logger.info("=" * 80)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load train.csv
    logger.info(f"Loading dataset from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    logger.info(f"Total series in dataset: {len(df)}")
    
    # Load error series IDs
    error_series_ids = set()
    if error_yaml_path and os.path.exists(error_yaml_path):
        error_series_ids = load_error_series_ids(error_yaml_path)
    
    # Filter out error series
    if error_series_ids:
        df_original_count = len(df)
        df = df[~df['SeriesInstanceUID'].isin(error_series_ids)]
        removed_count = df_original_count - len(df)
        logger.info(f"Removed {removed_count} error series")
        logger.info(f"Remaining series: {len(df)}")
    
    # Normalize modality names
    df['Modality_Normalized'] = df['Modality'].apply(normalize_modality)
    
    # Remove unknown modalities
    df = df[df['Modality_Normalized'] != 'Unknown']
    
    # Display modality distribution
    logger.info("\nOriginal Modality Distribution:")
    modality_counts = df['Modality_Normalized'].value_counts()
    for modality, count in modality_counts.items():
        logger.info(f"  {modality}: {count}")
    
    # Display label distribution
    logger.info("\nAneurysm Label Distribution:")
    label_counts = df['Aneurysm Present'].value_counts()
    for label, count in label_counts.items():
        label_name = "Aneurysm" if label == 1 else "No Aneurysm"
        logger.info(f"  {label_name}: {count}")
    
    # Selection targets
    targets = {
        'CTA': {
            'aneurysm': 500,
            'no_aneurysm': 500
        },
        'MRA': {
            'aneurysm': 250,
            'no_aneurysm': 250
        },
        'MRI_T1post': {
            'aneurysm': 125,
            'no_aneurysm': 125
        },
        'MRI_T2': {
            'aneurysm': 125,
            'no_aneurysm': 125
        }
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("TARGET DISTRIBUTION")
    logger.info("=" * 80)
    logger.info("CTA: 1000 total (500 aneurysm, 500 no aneurysm)")
    logger.info("MRA: 500 total (250 aneurysm, 250 no aneurysm)")
    logger.info("MRI: 500 total")
    logger.info("  - T1post: 250 (125 aneurysm, 125 no aneurysm)")
    logger.info("  - T2: 250 (125 aneurysm, 125 no aneurysm)")
    logger.info("TOTAL: 2000 series")
    logger.info("=" * 80)
    
    # Select samples for each modality and label
    selected_dfs = []
    
    for modality, label_targets in targets.items():
        logger.info(f"\nSelecting {modality} samples...")
        
        # Filter by modality
        modality_df = df[df['Modality_Normalized'] == modality].copy()
        
        for label_name, target_count in label_targets.items():
            # Determine label value
            label_value = 1 if label_name == 'aneurysm' else 0
            
            # Filter by label
            subset = modality_df[modality_df['Aneurysm Present'] == label_value]
            available_count = len(subset)
            
            # Check if enough samples available
            if available_count < target_count:
                logger.warning(
                    f"  {modality} {label_name}: Only {available_count} available "
                    f"(target: {target_count}). Using all available."
                )
                selected = subset
            else:
                # Randomly select target number of samples
                selected = subset.sample(n=target_count, random_state=random_seed)
                logger.info(
                    f"  {modality} {label_name}: Selected {len(selected)} / {available_count} available"
                )
            
            selected_dfs.append(selected)
    
    # Combine all selected samples
    selected_df = pd.concat(selected_dfs, ignore_index=True)
    
    # Shuffle the combined dataset
    selected_df = selected_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SELECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total selected series: {len(selected_df)}")
    
    logger.info("\nSelected Modality Distribution:")
    for modality, count in selected_df['Modality_Normalized'].value_counts().items():
        logger.info(f"  {modality}: {count}")
    
    logger.info("\nSelected Label Distribution:")
    for modality in selected_df['Modality_Normalized'].unique():
        modality_subset = selected_df[selected_df['Modality_Normalized'] == modality]
        aneurysm_count = len(modality_subset[modality_subset['Aneurysm Present'] == 1])
        no_aneurysm_count = len(modality_subset[modality_subset['Aneurysm Present'] == 0])
        logger.info(f"  {modality}:")
        logger.info(f"    - Aneurysm: {aneurysm_count}")
        logger.info(f"    - No Aneurysm: {no_aneurysm_count}")
    
    # Save to CSV if output path provided
    if output_csv_path:
        selected_df.to_csv(output_csv_path, index=False)
        logger.info(f"\nSelected dataset saved to: {output_csv_path}")
    
    logger.info("=" * 80)
    logger.info("DATASET SELECTION COMPLETED")
    logger.info("=" * 80)
    
    return selected_df


def create_train_val_test_splits(
    selected_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split selected dataset into train, validation, and test sets.
    Ensures balanced distribution across modalities and labels.
    
    Args:
        selected_df: DataFrame with selected series
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        output_dir: Directory to save split CSVs (optional)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAIN/VAL/TEST SPLITS")
    logger.info("=" * 80)
    
    np.random.seed(random_seed)
    
    # Stratified split by modality and label
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for modality in selected_df['Modality_Normalized'].unique():
        for label in [0, 1]:
            # Get subset for this modality-label combination
            subset = selected_df[
                (selected_df['Modality_Normalized'] == modality) &
                (selected_df['Aneurysm Present'] == label)
            ].copy()
            
            # Shuffle
            subset = subset.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # Calculate split indices
            n_total = len(subset)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split
            train_subset = subset.iloc[:n_train]
            val_subset = subset.iloc[n_train:n_train + n_val]
            test_subset = subset.iloc[n_train + n_val:]
            
            train_dfs.append(train_subset)
            val_dfs.append(val_subset)
            test_dfs.append(test_subset)
    
    # Combine and shuffle
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Summary
    logger.info(f"Training set: {len(train_df)} samples ({len(train_df) / len(selected_df) * 100:.1f}%)")
    logger.info(f"Validation set: {len(val_df)} samples ({len(val_df) / len(selected_df) * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df) / len(selected_df) * 100:.1f}%)")
    
    # Save splits if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_csv = os.path.join(output_dir, 'train_split.csv')
        val_csv = os.path.join(output_dir, 'val_split.csv')
        test_csv = os.path.join(output_dir, 'test_split.csv')
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        logger.info(f"\nSplits saved to: {output_dir}")
        logger.info(f"  - {train_csv}")
        logger.info(f"  - {val_csv}")
        logger.info(f"  - {test_csv}")
    
    logger.info("=" * 80)
    
    return train_df, val_df, test_df


# Main execution
if __name__ == "__main__":
    # Configuration
    TRAIN_CSV_PATH = r"D:\FYP_Data\rsna-intracranial-aneurysm-detection\train.csv"
    ERROR_YAML_PATH = r"D:\FYP_Data\rsna-intracranial-aneurysm-detection\error_data.yaml"
    OUTPUT_DIR = r"E:\Education\Aerux_Final\data_splits"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Select balanced dataset
    selected_df = select_balanced_dataset(
        train_csv_path=TRAIN_CSV_PATH,
        error_yaml_path=ERROR_YAML_PATH,
        output_csv_path=os.path.join(OUTPUT_DIR, 'selected_dataset_2000.csv'),
        random_seed=42
    )
    
    # Step 2: Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_splits(
        selected_df=selected_df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    print("DATASET SELECTION AND SPLITTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - selected_dataset_2000.csv: Full selected dataset")
    print(f"  - train_split.csv: Training set ({len(train_df)} samples)")
    print(f"  - val_split.csv: Validation set ({len(val_df)} samples)")
    print(f"  - test_split.csv: Test set ({len(test_df)} samples)")
    print("=" * 80)
