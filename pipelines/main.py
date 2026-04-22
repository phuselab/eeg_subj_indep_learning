import logging
from pathlib import Path
import torch
import numpy as np

from configs.config import create_config
from data import create_disjoint_loaders
from models.backbones import create_backbone
from models.disentanglement.core import DisentangledEEGModel
from models.losses import DisentanglementLoss
from pipelines.training import DVAETrainer
from utils.helper import setup_logging
import wandb
from utils.helper import compute_class_weights
from utils.argparser import apply_loss_weight_overrides
from torch import nn


def run_train_backbone_then_disentangle(args, config=None):
    """
    Runs the 'pretrained' two-stage training pipeline.
    This corresponds to the old 'Mode C'.
    """
    
    setup_logging(args.save_dir)
    # Setup also wandb logging here if needed
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args + config) if config is not None else vars(args),
            entity="wandbuser",
            group=args.run_name,
            name=f'{args.run_name}-BB_FT'
        )
    logging.info("Running 'Pretrained' Pipeline (formerly Mode C)")
    logging.info(f"Arguments: {vars(args)}\n")

    logging.info(f"Loading data from {args.data_file}...")
    data_dict = torch.load(args.data_file, weights_only=False)

    data_dict['data'] = data_dict.get('data', data_dict.get('X'))
    
    data_shape = (data_dict['data'].shape[1], data_dict['data'].shape[2])
    logging.info(f"Data shape: {data_dict['data'].shape} -> Input: {data_shape}")
    print("Excluding task ", args.exclude_tasks)
 
    # Write channel names of data_dict in config if they exist  
    ch_names = data_dict['ch_names'] if 'ch_names' in data_dict else None
    config.ch_names = ch_names
    
    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
        data_dict, args.batch_size, args.disjoint_split_ratio, exclude_tasks=[args.exclude_tasks]
    )
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"STFTLoss: Detected {num_devices} CUDA device(s). Using device: {torch.cuda.get_device_name(0)}")
    # select last GPU if multiple are available
    gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
    task_weights = compute_class_weights(train_loader, device=gpu)
    weights_dict = {'task': task_weights}
    
    config = config if config is not None else create_config(
        num_subjects=loader_info['train']['num_subjects'],
        num_tasks=loader_info['train']['num_tasks'],
        data_shape=data_shape, 
        mid_channels=args.mid_channels
    )
    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False, num_classes=loader_info['train']['num_tasks'])
    
    proj_out = backbone.model.proj_out if hasattr(backbone.model, 'proj_out') else None

    backbone.model.proj_out = nn.Identity()  # Ensure identity for feature extraction

    # reconstruction_backbone = create_backbone(args, data_shape, config=config)
    
    model = DisentangledEEGModel(backbone, config=config, phase_name='BB_FT', reconstruction_decoder=proj_out, classifier_type=args.classifier_type)
    model.to(gpu)
    
    WEIGHT_CLASSES = True
    if WEIGHT_CLASSES:
        # Initialize loss with weights
        loss_fn = DisentanglementLoss(config.loss_config_stage1, class_weights=weights_dict, segment_length=config.time_samples)
    else: # TODO until now the function is overridden!! 
        loss_fn = DisentanglementLoss(config.loss_config_stage1, segment_length=config.time_samples)


    # Optimizer with two lr for backbone and DVAE parts
    backbone_params = list(model.feature_extractor.parameters())
    dvae_params = []
    for name, module in model.named_children():
        if name != 'feature_extractor':
            dvae_params += [p for p in module.parameters() if p.requires_grad]
    


    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.lr},  # Lower LR for backbone
            {'params': dvae_params, 'lr': args.lr*5}             # Standard LR for DVAE
        ],
    )
        
    stage1_trainer = DVAETrainer(model, loss_fn, optimizer, save_dir=args.save_dir, phase_name="BB_FT", device=gpu)
    stage1_trainer.train(train_loader, val_loader, test_loader,args.stage1_epochs, wandb_run=wandb if args.use_wandb else None)
    config_dict = vars(config)
    config_vars = vars(args).copy()
    config_vars.update(config_dict)
    config_vars.update({'data_shape': data_shape})
    config_vars.update({'mid_channels': args.mid_channels})
   
    
    torch.save(config_vars, Path(args.save_dir) / 'run_config.pt')  

    if args.use_wandb:
        wandb.finish()

        wandb.init(
            project=args.project_name,
            config=config_vars,
            entity="wandbuser",
            group=args.run_name,
            name=f'{args.run_name}-DVAE'
        )

    model.set_phase('DVAE')
    # model.feature_extractor.freeze()

    optimizer_2 = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.lr},  # Lower LR for backbone
            {'params': dvae_params, 'lr': args.lr*5}             # Standard LR for DVAE
        ],
    )
    
    stage2_loss_config = apply_loss_weight_overrides(config.loss_config_stage2, args)
    
    if WEIGHT_CLASSES:
        #loss_fn = DisentanglementLoss(config.loss_config_stage2, discriminator=model.discriminator, class_weights=weights_dict)
        loss_fn = DisentanglementLoss(stage2_loss_config, discriminator=model.discriminator, class_weights=weights_dict, segment_length=config.time_samples)
    else:
        loss_fn = DisentanglementLoss(config.loss_config_stage2, discriminator=model.discriminator, segment_length=config.time_samples)
    stage2_trainer = DVAETrainer(model, loss_fn, optimizer_2, save_dir=args.save_dir, phase_name="DVAE", device=gpu)
    stage2_trainer.train(train_loader, val_loader, test_loader, args.epochs, wandb_run=wandb if args.use_wandb else None)
    
    # modifica modello 
    
    
    

