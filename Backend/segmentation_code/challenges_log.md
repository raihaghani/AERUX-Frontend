# 3D Segmentation Pipeline: Challenges & Solutions Log

This document tracks the technical hurdles encountered while transitioning the medical segmentation pipeline from 2.5D to full 3D, and how they were systematically resolved.

## 1. 8GB VRAM Constraint for 3D Segmentation

**Challenge**: Training 3D models on high-resolution medical volumes (e.g., 512x512x100) instantly throws "Out of Memory" (OOM) errors on consumer GPUs like the RTX 3070 (8GB).
**Solution**:

- Transitioned to **Patch-Based Training** using MONAI's `RandCropByPosNegLabeld` to extract memory-friendly `64x64x64` cubes.
- Scaled down the model complexity by initializing `SegResNet` with `init_filters=8`.
- Replaced large batch sizes with `batch_size=1` combined with **Gradient Accumulation** (4 steps) and Mixed Precision (`torch.amp`).

## 2. Model Import Versioning

**Challenge**: Attempted to load `HighRes3DNet`, but the installed version of MONAI used an older naming convention.
**Error Trace**:

```text
ImportError: cannot import name 'HighRes3DNet' from 'monai.networks.nets' (E:\Education\Aerux_Final\.venv\Lib\site-packages\monai\networks\nets\__init__.py). Did you mean: 'HighResNet'?
```

**Solution**: Corrected the import in `models.py` from `HighRes3DNet` to `HighResNet`.

## 3. Slow I/O and Missing `ITKReader`

**Challenge**: Training was hanging because loading thousands of raw DICOM files dynamically per epoch is extremely I/O intensive, especially without the compiled `itk` backend.
**Error Trace**:

```text
UserWarning: required package for reader ITKReader is not installed, or the version doesn't match requirement.
```

**Solution**:

- Created a utility script (`convert_dicom_to_nifti.py`) to pre-convert DICOM series folders into single compressed `.nii.gz` files.
- Updated `dataset.py` to point to the pre-converted `series_nifti` directory.

## 4. Corrupted DICOM Series in Dataset

**Challenge**: Certain DICOM series were inherently corrupted (varying slice thicknesses, mixed directions, or lacking annotations), causing fatal crashes.
**Solution**:

- Parsed the provided `error_data.yaml` dynamically in `dataset.py` and `convert_dicom_to_nifti.py`.
- Any `SeriesInstanceUID` listed in the YAML file is now automatically skipped during dataset scanning.

## 5. CUDA `device-side assert triggered` in DiceLoss

**Challenge**: PyTorch threw an index-out-of-bounds error during `loss_function(outputs, labels)`. This happened because the raw NIfTI segmentations contained voxel integer values greater than 1 (e.g., 255), which violated the 2-channel one-hot encoding requirement of MONAI's `DiceLoss`.
**Error Trace**:

```text
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\ScatterGatherKernel.cu:410: block: [2,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
torch.AcceleratorError: CUDA error: device-side assert triggered
```

**Solution**:

- Inserted a strict binarization check before the model forward pass: `labels = (labels > 0).long()`. This guarantees all background pixels are `0` and all aneurysm pixels are `1`.

## 6. `RandCrop` Dimension ValueError

**Challenge**: The dataset contained volumes with fewer slices than the crop size (e.g., a volume with a Z-dimension of only 24). `RandCropByPosNegLabeld(64, 64, 64)` threw a `ValueError` because it cannot crop 64 slices out of 24.
**Error Trace**:

```text
ValueError: The size of the proposed random crop ROI is larger than the image size, got ROI size (64, 64, 64) and label image size (512, 512, 24) respectively.
```

**Solution**:

- Integrated `SpatialPadd(spatial_size=(64, 64, 64))` into the preprocessing transforms. This ensures any dimension smaller than 64 is symmetrically padded with background (zeros) before the cropper reaches it.

## 7. PyTorch Conv3d Kernel Dimension Error

**Challenge**: Certain highly corrupted files yielded a tensor with a depth of `0` or `2`, causing the very first `3x3x3` convolution in `SegResNet` to crash due to the input being smaller than the filter.
**Error Trace**:

```text
RuntimeError: Calculated padded input size per channel: (66 x 66 x 2). Kernel size: (3 x 3 x 3). Kernel size can't be greater than actual input size
```

**Solution**:

- Implemented robust dynamic shape validation directly inside the training and validation loops:
  ```python
  if inputs.ndim != 5 or min(inputs.shape[2:]) < 3:
      print(f"Skipping batch due to invalid shape...")
      continue
  ```
- This allows the pipeline to safely step over "silent" corrupted batches without halting the entire multi-hour training run.

## 8. PyTorch DataLoader Stacking Error (Shape Mismatch)

**Challenge**: The DataLoader crashed while attempting to batch crops together because `RandCropByPosNegLabeld(num_samples=2)` generated crops of unequal sizes from the same volume (one was `64x64x64`, the other was `64x64x0` or `19x64x64`). This occurred because the `label` mask had a different number of slices than the `image` volume in the raw NIfTI files; the cropper targeted a center that existed in the label but fell off the edge of the image, leading to a truncated crop slice.
**Error Trace**:

```text
RuntimeError: stack expects each tensor to be equal size, but got [1, 64, 0, 64] at entry 0 and [1, 19, 64, 64] at entry 1
```

**Solution**:

- Initially added MONAI's `pad_list_data_collate`. However, this was insufficient because the extremely misaligned slices still caused problems.
- **Final Fix**: Implemented a custom MONAI `MapTransform` class called `MatchLabelToImage`. This transform is applied immediately after the NIfTI files are loaded. It actively compares the shape of the `label` against the `image`, and mathematically crops or pads the `label` array using `torch.nn.functional.pad` to guarantee their spatial dimensions match exactly before any downstream croppers or augmentations touch them.

## 9. MONAI RandCrop Broadcasting Error (Label and Image Shape Mismatch)

**Challenge**: The dataset contained image/label pairs with slightly mismatched Z-dimensions (e.g., the scan had 177 slices, but the label had 176 slices). When `RandCropByPosNegLabeld` attempted to combine them to find valid bounding boxes using `image_threshold=0`, PyTorch threw a broadcasting crash because it tried to run a bitwise `&` operator between arrays of different sizes.
**Error Trace**:

```text
RuntimeError: The size of tensor a (46399488) must match the size of tensor b (46137344) at non-singleton dimension 0
```

**Solution**:

- Removed the `image_key="image"` and `image_threshold=0` arguments from `RandCropByPosNegLabeld` in `dataset.py`. This prevents MONAI from attempting to compute the intersection between the image and label arrays for bounding box limits, bypassing the broadcasting error and allowing it to crop based purely on the label coordinates. (Any out-of-bounds edge slices will be safely zero-padded by our previously implemented `pad_list_data_collate`).

## 10. Validation Loop CUDA OOM

**Challenge**: Training successfully completed all steps, but immediately crashed with an "Out of Memory" error the moment the validation loop began. Inference over full 3D volumes (e.g., `512x512x150`) using sliding windows requires significantly more contiguous memory than training on small `64x64x64` patches.
**Error Trace**:

```text
torch.AcceleratorError: Caught AcceleratorError in pin memory thread for device 0.
torch.AcceleratorError: CUDA error: out of memory
```

**Solution**:

- **Disabled `pin_memory`**: Set `pin_memory=False` on the `train_loader` and `val_loader` in `dataset.py` to prevent background threads from attempting aggressive memory allocation.
- **Cleared Cache**: Inserted `torch.cuda.empty_cache()` immediately before the validation loop in `train_3d.py` to completely defragment and release training VRAM before large-volume inference begins.
- **Reduced Sliding Window Batch**: Lowered `sw_batch_size` from 2 down to 1 inside `sliding_window_inference` to minimize the number of patches processed concurrently during validation.

## 11. Graceful Recovery and Resuming Training

**Challenge**: Training could be interrupted due to manual stops (Ctrl+C) or external crashes, risking the loss of epoch progress.
**Solution**:

- Added a `checkpoint_segresnet.pth` system to the end of every epoch that saves the model weights, optimizer state, scaler state, and current epoch count.
- Upon startup, the script checks for this file and seamlessly restores the training loop from exactly where it left off.
