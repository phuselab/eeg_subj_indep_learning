import argparse
import logging
from enum import Enum



def add_loss_weight_args(parser):
    """Add stage-2 loss weight overrides to the argument parser."""
    g = parser.add_argument_group("Stage-2 Loss Weights (grid search overrides)")

    g.add_argument("--loss-reconstruction-weight", type=float, default=None)
    g.add_argument("--loss-reconstruction-weight-mse", type=float, default=None)
    
    g.add_argument("--loss-kl-weight", type=float, default=None)
    g.add_argument("--loss-noise-kl-weight", type=float, default=None)
    
    g.add_argument("--loss-class-weight", type=float, default=None)
    g.add_argument("--loss-var-class-weight", type=float, default=None)
    
    g.add_argument("--loss-self-cycle-weight", type=float, default=None)
    g.add_argument("--loss-intra-weight", type=float, default=None)
    g.add_argument("--loss-cross-class-weight", type=float, default=None)
    g.add_argument("--loss-cross-cycle-weight", type=float, default=None)
    
    g.add_argument("--loss-kd-weight", type=float, default=None)
    g.add_argument("--loss-adv-weight", type=float, default=None)
    
def apply_loss_weight_overrides(loss_config, args):
    overrides = {
        "self_reconstruction_weight": args.loss_reconstruction_weight,
        "self_reconstruction_weight_mse": args.loss_reconstruction_weight_mse,  
        
        "kl_weight": args.loss_kl_weight,
        "noise_kl_weight": args.loss_noise_kl_weight,
        
        "classification_weight": args.loss_class_weight,
        "var_classification_weight": args.loss_var_class_weight,
        
        "self_cycle_weight": args.loss_self_cycle_weight, 
        "cross_subject_intra_class_weight": args.loss_intra_weight,
        "cross_subject_cross_class_weight": args.loss_cross_class_weight,
        "cross_cross_cycle_weight": args.loss_cross_cycle_weight,
        
        "kd_weight": args.loss_kd_weight,
        "adversarial_weight": args.loss_adv_weight,
    }

    for field, value in overrides.items():
        if value is not None:
            setattr(loss_config, field, value)
            logging.info(f"[LossOverride] {field} = {value}")
    return loss_config



def apply_mid_channels_override(config, args):
    """Override mid_channels in config if specified in args."""
    if args.mid_channels is not None and len(args.mid_channels) > 0:
        config.mid_channels = args.mid_channels
        logging.info(f"[ConfigOverride] mid_channels = {args.mid_channels}")
    return config

def parse_training_args():
    """Enhanced argument parser with clear mode selection"""
    parser = argparse.ArgumentParser(description='Modular DVAE Training')
    
    # Data arguments
    parser.add_argument('--data-file', required=True,
                       help='Path to .pt file (raw data or pre-extracted features)')
    
    parser.add_argument('--analysis-block', type=str, required=True, choices=['lbm', 'alignment', 'disentanglement'], help='Which analysis block to run')
    parser.add_argument('--yaml-config', type=str, help='Path to YAML config file for training parameters (when training a DVAE)')
    # Backbone arguments (required for modes B, C, D)
    parser.add_argument('--backbone', choices=['labram', 'cbramod', 'signal_processing', 'eegpt'],
                       help='Backbone architecture (required for modes B/C/D)')
    parser.add_argument('--backbone-weights',
                       help='Path to backbone weights (required for modes B/C/D)')
    parser.add_argument('--xgb-classifier', type=int, default=1, help='Whether to use an XGBoost classifier for evaluating the quality of the latent spaces during training (default: True)')
    parser.add_argument('--tsne', type=int, default=1, help='Whether to perform t-SNE visualization of latent spaces during training (default: True)')
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--project-name', type=str, default='"CBraMod_finetune_MI"',
                       help='WandB project name for logging')
    
    # Mode-specific arguments
    parser.add_argument('--stage1-epochs', type=int, default=10,
                       help='[Mode C] Epochs for backbone adaptation stage')
    parser.add_argument('--disjoint-split-ratio', type=float, default=0.3,
                       help='[Mode C] Ratio of train data for T_A')
    parser.add_argument('--freeze-after-epoch', type=int, default=20,
                       help='[Mode D] Epoch to freeze backbone')
    #parser.add_argument('--include-subject-classifiers', action='store_true',
    #                   help='[Mode B/C] Whether to include subject classifiers during Stage 1 finetuning')
    
    # Projector architecture (Mode A)
    parser.add_argument('--projector-hidden-dim', type=int, default=512,
                       help='[Mode A] Hidden dimension for projector')
    parser.add_argument('--projector-dropout', type=float, default=0.1,
                       help='[Mode A] Dropout rate for projector')
    
    # Include subject classifier for mode C if desired, set default to True 
    parser.add_argument('--include-subject-classifiers', action='store_true', default=True,
                       help='[Mode C] Include subject classifier during Stage 1 finetuning (default: True)')     
    
    # Add phase argument
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Training phase for the encoders to stabiölize training: 1=Task only, 2=Task+Subject, 3=Full model (also noise encoder and adversarial)')
    parser.add_argument('--exclude_tasks', type=int, default=None, help="Whether there is a task to exclude")
    parser.add_argument('--exclude-subjects', nargs='*', type=int, default=[], help="List of subject IDs to exclude from training and evaluation")
    parser.add_argument('--skip-backbone', type=int, default=0, help='Whether to skip backbone training and only train the classifier head (default: False)')
    parser.add_argument('--classifier-type', type=str, default='diva_classifier', choices=['diva_classifier', 'cbramod_classifier', 'labram_classifier', 'eegpt_classifier'], help='Type of classifier head to use (default: diva_classifier)')
    # parser.add_argument('--alignment-type', type=str, default=None, choices= [None, 'euclideanAlignment', 'latentAlignment2d', 'adaptiveBatchNorm', 'batchNorm'], help='Type of alignment loss to use (default: None)')
    parser.add_argument('--mid_channels', nargs='*', type=int, default=[64, 32, 16, 8], help='List of channel sizes for the feature extractor (default: [64, 32, 16, 8])')
    parser.add_argument('--divisor', type=float, default=100.0, help='Divisor for scaling data range, default 100.0 to adjust eeg range')
    # Logging and saving
    parser.add_argument('--run-name', type=str, default='',
                       help='Name for the training run (in WandB)', required=True)
    
    parser.add_argument('--use-wandb', action='store_true',
                       help='Whether to use Weights and Biases for logging')

    parser.add_argument('--save-dir', required=True)
    add_loss_weight_args(parser) 
    
    return parser.parse_args()


# def validate_args(args):
#     """Validate argument combinations"""
#     mode = get_training_mode(args)
    
#     # Modes B, C, D require backbone specification
#     # Im mode F is chosen, then it is not a dl backbone
#     is_dl_backbone = mode != TrainingMode.SIGNAL_FEATURES
    
#     # Modes B, C, D require backbone specification
#     if mode in [TrainingMode.FROZEN_BACKBONE, TrainingMode.TWO_STAGE_DISJOINT, 
#                 TrainingMode.JOINT_THEN_FROZEN, TrainingMode.CLASSIFIER_ONLY]:
#         if not args.backbone:
#             raise ValueError(f"{mode.value} requires --backbone")
#         # We need weights only for DL backbones
#         if is_dl_backbone and not args.backbone_weights:
#             raise ValueError(f"{mode.value} requires --backbone-weights")
    
#     # Mode A and E shouldn't specify backbone
#     if mode == TrainingMode.RAW_LEARNABLE: # or mode == TrainingMode.SIGNAL_FEATURES:
#         if args.backbone or args.backbone_weights:
#             logging.warning("Mode A and E ignores --backbone and --backbone-weights")
    
#     # Validate file paths exist
#     from pathlib import Path
#     if not Path(args.data_file).exists():
#         raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
#     if args.backbone_weights and is_dl_backbone and not Path(args.backbone_weights).exists():
#         raise FileNotFoundError(f"Backbone weights not found: {args.backbone_weights}")
