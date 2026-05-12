import os
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    SpatialPadd,
    ResizeWithPadOrCropd,
)
from monai.data import pad_list_data_collate
from monai.transforms import MapTransform
from sklearn.model_selection import train_test_split
import torch

class MatchLabelToImage(MapTransform):
    """Dynamically crops/pads the label to exactly match the image shape."""
    def __init__(self, keys, image_key="image", label_key="label"):
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        lbl = d[self.label_key]
        
        if img.shape == lbl.shape:
            return d
            
        # Crop label if it is larger than image
        slices = [slice(None)] * lbl.ndim
        for i in range(1, lbl.ndim):
            slices[i] = slice(0, min(lbl.shape[i], img.shape[i]))
        lbl = lbl[tuple(slices)]
        
        # Pad label if it is smaller than image
        pad_size = []
        for i in reversed(range(1, lbl.ndim)):
            diff = max(0, img.shape[i] - lbl.shape[i])
            pad_size.extend([0, diff])
            
        if sum(pad_size) > 0:
            lbl = torch.nn.functional.pad(lbl, pad_size, mode='constant', value=0)
            
        d[self.label_key] = lbl
        return d

def get_dataloaders(data_dir, batch_size=1, val_split=0.2):
    """
    Parses the dataset directory and returns MONAI DataLoaders for 3D training.
    """
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find train.csv at {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Load error UIDs to skip
    error_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "error_data.yaml")
    error_uids = set()
    if os.path.exists(error_yaml):
        with open(error_yaml, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('- '):
                    uid = line[2:].split('#')[0].strip()
                    if uid:
                        error_uids.add(uid)
    print(f"Loaded {len(error_uids)} error UIDs to skip.")
    
    # Only keep those with 'Aneurysm Present' if we only want to train on positive samples
    # Or train on all. For segmentation, usually we want all or at least those with segmentations.
    # We will just verify the files exist.
    data_dicts = []
    
    print("Scanning dataset to map DICOM series and NIfTI segmentations...")
    for idx, row in df.iterrows():
        sid = str(row['SeriesInstanceUID'])
        
        if sid in error_uids:
            continue
            
        image_dir = os.path.join(data_dir, "series_nifti", f"{sid}.nii.gz")
        # Using the original segmentation mask
        label_path = os.path.join(data_dir, "segmentations", f"{sid}.nii")
        
        if os.path.exists(image_dir) and os.path.exists(label_path):
            data_dicts.append({
                "image": image_dir,
                "label": label_path
            })
            
    print(f"Found {len(data_dicts)} valid paired 3D samples.")
    
    if len(data_dicts) == 0:
        raise ValueError("No valid image-label pairs found. Check data paths.")

    train_files, val_files = train_test_split(data_dicts, test_size=val_split, random_state=42)

    # Transform pipeline for training
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="ITKReader"), # ITK natively loads DICOM directories
            EnsureChannelFirstd(keys=["image", "label"]),
            
            # CRITICAL FIX: Ensure label perfectly matches image spatial dimensions before any padding/cropping
            MatchLabelToImage(keys=["label"]),
            
            # (Optional) Spacing: resamples image to consistent voxel sizes if needed
            # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            
            # Normalize intensity (Assuming CTA/MRA, usually -100 to 300 or similar for vessels)
            # You might need to adjust these values based on your specific modality
            ScaleIntensityRanged(
                keys=["image"], a_min=-100, a_max=300,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            
            # Ensure the image is at least 64x64x64 so the cropper doesn't fail on smaller images
            SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
            
            # CRITICAL FOR 8GB VRAM: Extract 64x64x64 patches
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
                pos=1,
                neg=1,
                num_samples=2, # Extracts 2 patches per full volume per epoch
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            
            # BULLETPROOF FIX: Force exactly 64x64x64 before batching
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(64, 64, 64)),
        ]
    )

    # Transform pipeline for validation
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            MatchLabelToImage(keys=["label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-100, a_max=300,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # No RandCrop during validation! We use sliding_window_inference in train.py
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, 
        pin_memory=False,
        collate_fn=pad_list_data_collate
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, 
        pin_memory=False,
        collate_fn=pad_list_data_collate
    )

    return train_loader, val_loader
