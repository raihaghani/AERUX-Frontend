import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize NIfTI image and predicted mask")
    parser.add_argument("--image", type=str, required=True, help="Path to the original NIfTI image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the predicted NIfTI mask")
    parser.add_argument("--slice_axis", type=int, default=2, choices=[0, 1, 2], help="Axis to slice along (0=Sagittal, 1=Coronal, 2=Axial)")
    args = parser.parse_args()

    # Load NIfTI files
    print(f"Loading image: {args.image}")
    img_nii = nib.load(args.image)
    img_data = img_nii.get_fdata()

    print(f"Loading mask: {args.mask}")
    mask_nii = nib.load(args.mask)
    mask_data = mask_nii.get_fdata()

    # Squeeze out single-dimension channels if present (e.g. [1, H, W, D])
    img_data = np.squeeze(img_data)
    mask_data = np.squeeze(mask_data)

    if img_data.shape != mask_data.shape:
        print(f"Warning: Image shape {img_data.shape} does not match mask shape {mask_data.shape}!")

    # Find slices where the mask actually has an aneurysm predicted (mask > 0)
    # This prevents scrolling through hundreds of empty background slices
    mask_indices = np.where(mask_data > 0)
    
    if len(mask_indices[0]) == 0:
        print("\nNo aneurysm predicted (mask is entirely empty/background).")
        print("Showing middle slice of the volume instead.")
        slice_idx = img_data.shape[args.slice_axis] // 2
        plot_slices(img_data, mask_data, [slice_idx], args.slice_axis)
        return

    # Find unique slices along the chosen axis that contain the predicted aneurysm
    valid_slices = np.unique(mask_indices[args.slice_axis])
    print(f"\nAneurysm predicted on {len(valid_slices)} slices along axis {args.slice_axis}.")
    
    # Pick up to 5 evenly spaced slices from the valid ones to visualize
    num_plots = min(5, len(valid_slices))
    selected_slices = np.linspace(0, len(valid_slices) - 1, num_plots, dtype=int)
    plot_indices = valid_slices[selected_slices]

    plot_slices(img_data, mask_data, plot_indices, args.slice_axis)

def plot_slices(img_data, mask_data, slice_indices, axis):
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 5 * len(slice_indices)))
    
    # Handle single slice case (1D array of axes)
    if len(slice_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(slice_indices):
        # Extract the 2D slice based on the chosen axis
        if axis == 0:
            img_slice = img_data[idx, :, :]
            mask_slice = mask_data[idx, :, :]
        elif axis == 1:
            img_slice = img_data[:, idx, :]
            mask_slice = mask_data[:, idx, :]
        else:
            img_slice = img_data[:, :, idx]
            mask_slice = mask_data[:, :, idx]

        # Rotate slices 90 degrees for better viewing orientation
        img_slice = np.rot90(img_slice)
        mask_slice = np.rot90(mask_slice)

        # 1. Original Image
        axes[i][0].imshow(img_slice, cmap='gray')
        axes[i][0].set_title(f"Original Image (Slice {idx})")
        axes[i][0].axis('off')

        # 2. Predicted Mask
        axes[i][1].imshow(mask_slice, cmap='inferno')
        axes[i][1].set_title(f"Predicted Mask")
        axes[i][1].axis('off')

        # 3. Overlay (Image + Mask)
        axes[i][2].imshow(img_slice, cmap='gray')
        
        # Create an alpha channel for the mask overlay: transparent where 0, solid red where 1
        mask_overlay = np.zeros((*mask_slice.shape, 4))
        mask_overlay[mask_slice > 0] = [1, 0, 0, 0.6] # Red with 60% opacity
        
        axes[i][2].imshow(mask_overlay)
        axes[i][2].set_title(f"Overlay")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
