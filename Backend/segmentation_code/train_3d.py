import os
import argparse
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from dataset import get_dataloaders
from models import get_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup DataLoaders
    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, val_split=0.2)
    
    # 2. Setup Model
    model = get_model(args.model, in_channels=1, out_channels=2).to(device)
    
    # 3. Loss and Optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # 4. Training Setup (Mixed Precision & Grad Accumulation)
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = args.grad_accum
    
    best_metric = -1
    best_metric_epoch = -1
    save_dir = os.path.join("weights")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{args.model}.pth")
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{args.model}.pth")

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
        print(f"Resuming from epoch {start_epoch + 1} with best validation metric {best_metric:.4f}")

    print(f"Starting training for {args.model} over {args.epochs} epochs.")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Ensure labels are strictly 0 or 1 (binary) to prevent CUDA out-of-bounds assert in to_onehot_y
            labels = (labels > 0).long()
            
            # Robustness: Skip corrupted batches that don't have the proper 3D shape
            # SegResNet requires 5D tensors [B, C, D, H, W] where spatial dims >= kernel_size (3)
            if inputs.ndim != 5 or min(inputs.shape[2:]) < 3:
                print(f"\n[WARNING] Skipping batch {step} due to invalid shape: {inputs.shape}")
                continue
            
            # Forward pass with Mixed Precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                # Normalize loss for accumulation
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accumulation_steps
            
            # Update weights after accumulation_steps
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            print(f"Epoch {epoch+1}/{args.epochs} Step {step}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}")
            
        epoch_loss /= step
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
        
        # 5. Validation Loop
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            
            # CRITICAL FOR 8GB VRAM: Clear cache before sliding_window_inference
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # Robustness: Skip corrupted validation batches
                    if val_inputs.ndim != 5 or min(val_inputs.shape[2:]) < 3:
                        print(f"\n[WARNING] Skipping validation batch due to invalid shape: {val_inputs.shape}")
                        continue
                        
                    # Ensure val_labels are strictly 0 or 1
                    val_labels = (val_labels > 0).long()
                    
                    # Inference over full volume using sliding window
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            val_inputs, 
                            roi_size=(64, 64, 64), 
                            sw_batch_size=1, # CRITICAL: Keep at 1 for 8GB VRAM
                            predictor=model,
                            overlap=0.5
                        )
                    
                    # Convert to onehot for metric
                    val_outputs = [torch.argmax(i, dim=0, keepdim=True) for i in val_outputs]
                    val_outputs = torch.stack(val_outputs)
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f"Validation Epoch {epoch + 1} - Mean Dice: {metric:.4f}")
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved new best metric model to {save_path}")
        
        # Save latest checkpoint for resuming at the end of every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
        }, checkpoint_path)
                    
    print(f"Training completed. Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight 3D Segmentation Training")
    parser.add_argument("--data_dir", type=str, default="G:\\FYP_Data\\rsna-intracranial-aneurysm-detection", help="Path to dataset")
    parser.add_argument("--model", type=str, choices=["segresnet", "highresnet"], default="segresnet", help="Model to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (Keep at 1 or 2 for 8GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval")
    
    args = parser.parse_args()
    main(args)
