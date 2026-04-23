import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional
from features.signal_processing import SignalProcessingBackbone
from .pretrained_backbones import LabramBackbone, CBraModBackbone, EEGPTBackbone
from configs.config import ModelConfig

def create_backbone(args, data_shape: Tuple[int, int], config: ModelConfig, use_identity_for_reconstruction: bool=True, num_classes: int = None) -> Tuple[Optional[nn.Module], int, bool]:
    """
    Create appropriate backbone based on training mode.
    
    Returns:
        backbone: Feature extractor module (or None for Mode A)
        feature_dim: Output feature dimension
        requires_grad: Whether backbone should be trainable initially
    """
    training_mode = getattr(args, 'training_mode', 'default') # safetey guard
    n_chans, n_times = data_shape
    print(f"Creating backbone for mode: {training_mode}, data shape: {data_shape}, n_chans: {n_chans}, n_times: {n_times}")
    
    # Signal Processing Features
    if training_mode == 'signal_features':
        sfreq = 128.0 # should not be hard coded
        
        backbone = SignalProcessingBackbone(
            n_chans=n_chans, n_times=n_times, sfreq=sfreq
        )
        # Feature dimension determined on first forward pass. Use placeholder 200 initially.
        dummy_input = torch.randn(1, n_chans, n_times, dtype=torch.float32)
        # forward pass to determine feature dim
        with torch.no_grad():
            dummy_output = backbone(dummy_input)
        if len(dummy_output.shape) == 2:
            feature_dim = dummy_output.shape[1]
        else:
            raise ValueError(f"SignalProcessingBackbone output unexpected shape: {dummy_output.shape}. Expected (B, F).")
        
        requires_grad = False # Fixed, non-learnable feature extraction
        logging.info(f"Created fixed signal processing backbone (Input: {n_chans}x{n_times})")
        return backbone, feature_dim, requires_grad
    
    # Pretrained Backbone
    else:
        patch_size = config.patch_size
        
        # Create backbone
        if args.backbone == 'labram':
            ch_names = config.ch_names
            backbone = LabramBackbone(
                n_chans=n_chans,
                n_times=n_times,
                weights_path=args.backbone_weights,
                patch_size=patch_size,
                emb_size=200,
                ch_names=ch_names
            )
        elif args.backbone == 'cbramod':
            in_dim = 200
            backbone = CBraModBackbone(
                in_dim=in_dim,
                time_samples=n_times,
                patch_size=patch_size,
                weights_path=args.backbone_weights,
                emb_dim=patch_size,
                use_identity_for_reconstruction=use_identity_for_reconstruction
            )
        elif args.backbone == 'eegpt':
            backbone = EEGPTBackbone(
                n_chans=n_chans,
                n_times=n_times,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown backbone: {args.backbone}")
        
        logging.info(f"Created {args.backbone} backbone")
    
    return backbone
