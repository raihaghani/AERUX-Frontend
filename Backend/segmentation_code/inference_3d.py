import argparse
import os
import torch
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    SaveImaged,
)
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset
from monai.handlers.utils import from_engine

def main():
    parser = argparse.ArgumentParser(description="3D Inference for Aneurysm Segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to the input NIfTI image (.nii or .nii.gz)")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Directory to save the predicted mask")
    parser.add_argument("--checkpoint", type=str, default="weights/checkpoint_segresnet.pth", help="Path to the trained model checkpoint")
    args = parser.parse_args()

    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")

    # 2. Define the Model (must match the training architecture exactly)
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2, # Binary: Background vs Aneurysm
        init_filters=8, # Scaled down for 8GB VRAM
    ).to(device)

    # Load weights
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Setup transforms
    # We only need basic loading and intensity scaling for inference
    infer_transforms = Compose(
        [
            LoadImaged(keys=["image"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-100, a_max=300,
                b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )

    # 4. Prepare data loader
    # We package the single input file into a dictionary
    files = [{"image": args.input}]
    infer_ds = Dataset(data=files, transform=infer_transforms)
    infer_loader = DataLoader(infer_ds, batch_size=1, num_workers=0)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Inference
    print(f"Starting inference on {args.input}...")
    with torch.no_grad():
        for infer_data in infer_loader:
            inputs = infer_data["image"].to(device)

            # Use sliding window inference to handle arbitrarily large volumes
            with torch.amp.autocast('cuda'):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=(64, 64, 64),
                    sw_batch_size=1, # Very safe for 8GB VRAM
                    predictor=model,
                    overlap=0.5
                )

            # Convert raw logits to a binary mask (0 for background, 1 for aneurysm)
            # outputs shape: [1, 2, H, W, D] -> [1, 1, H, W, D]
            pred_mask = torch.argmax(outputs, dim=1, keepdim=True)
            
            # Pack it back into the dictionary for MONAI's SaveImaged
            infer_data["pred"] = pred_mask

            # 6. Save the prediction
            # SaveImaged will automatically match the spatial affine and metadata of the input image!
            saver = SaveImaged(
                keys="pred",
                output_dir=args.output_dir,
                output_postfix="seg",
                resample=False, # We want the mask to perfectly overlap the original input voxels
                output_ext=".nii.gz",
                print_log=True
            )
            
            # SaveImaged expects a dictionary or list of dictionaries
            # Since pred_mask is batched (B=1), we slice out the first element
            # But SaveImaged can handle batched dicts if used carefully. 
            # The cleanest way is to just call it on the dictionary, but we have to ensure meta_dict is present
            
            # Actually, to make it perfectly robust with MONAI's meta-tensors:
            infer_data["pred"] = pred_mask[0] # remove batch dim
            infer_data["image"] = infer_data["image"][0] # remove batch dim
            
            # If the original image was loaded as a MetaTensor, its affine is preserved
            saver(infer_data)

    print(f"\nInference complete! Saved segmentation mask to {args.output_dir}")

if __name__ == "__main__":
    main()
