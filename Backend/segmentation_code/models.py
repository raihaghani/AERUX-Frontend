from monai.networks.nets import SegResNet, HighResNet

def get_model(model_name="segresnet", in_channels=1, out_channels=2):
    """
    Returns a lightweight 3D segmentation model suitable for 8GB VRAM.
    """
    model_name = model_name.lower()
    
    if model_name == "segresnet":
        # SegResNet is highly efficient if we drastically reduce init_filters
        # Default init_filters=32 requires ~24GB VRAM for 3D. 
        # init_filters=8 uses a fraction of the memory while maintaining accuracy.
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=8,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )
    elif model_name == "highresnet":
        # HighRes3DNet is natively very compact because it relies on dilated 
        # convolutions rather than heavy upsampling/downsampling feature maps.
        model = HighResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'segresnet' or 'highresnet'.")
        
    return model
