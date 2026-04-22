import logging
from pathlib import Path
import torch
import numpy as np

# from configs.config import create_config
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
from configs.config import load_model_config, update_tasks_subjects


def run_cbramod_diva(args, config=None):
    setup_logging(args.save_dir)
    # Setup also wandb logging here if needed
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args + config) if config is not None else vars(args),
            entity="giuseppe-facchi-phuselab",
            group=args.run_name,
            name=f'{args.run_name}-BB_FT'
        )
    logging.info("Running 'Pretrained' Pipeline")
    logging.info(f"Arguments: {vars(args)}\n")

    logging.info(f"Loading data from {args.data_file}...")
    data_dict = torch.load(args.data_file, weights_only=False)

    data_dict['data'] = data_dict.get('data', data_dict.get('X'))
    
    data_shape = (data_dict['data'].shape[1], data_dict['data'].shape[2])
    logging.info(f"Data shape: {data_dict['data'].shape} -> Input: {data_shape}")
    print("Excluding task ", args.exclude_tasks)
 
    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
        data_dict, args.batch_size, args.disjoint_split_ratio, 
        exclude_tasks=[args.exclude_tasks], divisor=args.divisor, exclude_subjects=args.exclude_subjects
    )
    # select last GPU if multiple are available
    gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
    task_weights = compute_class_weights(train_loader, device=gpu)
    weights_dict = {'task': task_weights}

    config = load_model_config(
        yaml_path=args.yaml_config, 
        data_shape=data_shape, 
        num_subjects=loader_info['train']['num_subjects'], 
        num_tasks=loader_info['train']['num_tasks']
    )

    config.data_shape = data_shape
    config.mid_channels = args.mid_channels

    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False, num_classes=loader_info['train']['num_tasks'])
    proj_out = backbone.model.proj_out if hasattr(backbone.model, 'proj_out') else None
    backbone.model.proj_out = nn.Identity()  # Ensure identity for feature extraction

    model = DisentangledEEGModel(backbone, config=config, phase_name='BB_FT', reconstruction_decoder=proj_out, classifier_type=args.classifier_type, skip_backbone=args.skip_backbone)
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
        
    stage1_trainer = DVAETrainer(model, loss_fn, optimizer, save_dir=args.save_dir, phase_name="BB_FT", device=gpu, xgb_classifier=args.xgb_classifier, tsne=args.tsne)
    stage1_trainer.train(train_loader, val_loader, test_loader,args.stage1_epochs, wandb_run=wandb if args.use_wandb else None)

    config_vars = vars(args).copy()
    
    torch.save(config_vars, Path(args.save_dir) / 'run_config.pt')  

    if args.use_wandb:
        wandb.finish()

        wandb.init(
            project=args.project_name,
            config=config_vars,
            entity="giuseppe-facchi-phuselab",
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
    stage2_trainer = DVAETrainer(model, loss_fn, optimizer_2, save_dir=args.save_dir, phase_name="DVAE", device=gpu, xgb_classifier=args.xgb_classifier, tsne=args.tsne)
    stage2_trainer.train(train_loader, val_loader, test_loader, args.epochs, wandb_run=wandb if args.use_wandb else None)
    


def run_lbm(args, config=None):
    logging.info("Running LBM Pipeline")
    setup_logging(args.save_dir)
    # Setup also wandb logging here if needed
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args + config) if config is not None else vars(args),
            entity="giuseppe-facchi-phuselab",
            group=args.run_name,
            name=f'{args.run_name}-LBM'
        )
    logging.info(f"Arguments: {vars(args)}\n")

    logging.info(f"Loading data from {args.data_file}...")
    data_dict = torch.load(args.data_file, weights_only=False)

    data_dict['data'] = data_dict.get('data', data_dict.get('X'))
    
    data_shape = (data_dict['data'].shape[1], data_dict['data'].shape[2])
    logging.info(f"Data shape: {data_dict['data'].shape} -> Input: {data_shape}")
    print("Excluding task ", args.exclude_tasks)
    
    ch_names = data_dict['ch_names'] if 'ch_names' in data_dict else None

    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
        data_dict, args.batch_size, args.disjoint_split_ratio, 
        exclude_tasks=[args.exclude_tasks], divisor=args.divisor, exclude_subjects=args.exclude_subjects
    )
    # select last GPU if multiple are available
    gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
    task_weights = compute_class_weights(train_loader, device=gpu)
    weights_dict = {'task': task_weights}

    config = load_model_config(
        yaml_path=args.yaml_config, 
        data_shape=data_shape, 
        num_subjects=loader_info['train']['num_subjects'], 
        num_tasks=loader_info['train']['num_tasks']
    )
    config.ch_names = ch_names



    config.data_shape = data_shape
    config.mid_channels = args.mid_channels

    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False, num_classes=loader_info['train']['num_tasks'])
    proj_out = backbone.model.proj_out if hasattr(backbone.model, 'proj_out') else None
    backbone.model.proj_out = nn.Identity()  # Ensure identity for feature extraction
    
    print("Args classifier type: ", args.classifier_type)
    model = DisentangledEEGModel(backbone, config=config, phase_name='BB_FT', reconstruction_decoder=proj_out, classifier_type=args.classifier_type, skip_backbone=args.skip_backbone)
    model.to(gpu)

    backbone_freeze = config.backbone_freeze

    model.feature_extractor.freeze() if backbone_freeze else model.feature_extractor.unfreeze()

    print("-" * 50)
    print("\n\nBackbone freeze: ", backbone_freeze, "\n\n")
    print("-" * 50)
    
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
        
    stage1_trainer = DVAETrainer(model, loss_fn, optimizer, save_dir=args.save_dir, phase_name="BB_FT", device=gpu, xgb_classifier=args.xgb_classifier, tsne=args.tsne)
    stage1_trainer.train(train_loader, val_loader, test_loader,args.stage1_epochs, wandb_run=wandb if args.use_wandb else None)

    config_vars = vars(args).copy()
    
    torch.save(config_vars, Path(args.save_dir) / 'run_config.pt')  

    if args.use_wandb:
        wandb.finish()



def run_alignment(args, config=None):
    logging.info("Running lbm with alignment pipeline")
    setup_logging(args.save_dir)
    # Setup also wandb logging here if needed
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args + config) if config is not None else vars(args),
            entity="giuseppe-facchi-phuselab",
            group=args.run_name,
            name=f'{args.run_name}-ALIGNMENT'
        )
    logging.info(f"Arguments: {vars(args)}\n")

    logging.info(f"Loading data from {args.data_file}...")
    data_dict = torch.load(args.data_file, weights_only=False)

    data_dict['data'] = data_dict.get('data', data_dict.get('X'))
    
    data_shape = (data_dict['data'].shape[1], data_dict['data'].shape[2])
    logging.info(f"Data shape: {data_dict['data'].shape} -> Input: {data_shape}")
    print("Excluding task ", args.exclude_tasks)
    
    config = load_model_config(
        yaml_path=args.yaml_config, 
        data_shape=data_shape, 
    )

    # Write channel names of data_dict in config if they exist  
    ch_names = data_dict['ch_names'] if 'ch_names' in data_dict else None
    config.ch_names = ch_names

    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
        data_dict, args.batch_size, args.disjoint_split_ratio, exclude_tasks=[args.exclude_tasks],
        by_subject=config.by_subject, by_subject_inference=config.by_subject_inference,
        divisor=args.divisor, exclude_subjects=args.exclude_subjects
    )

    config = update_tasks_subjects(config, num_tasks=loader_info['train']['num_tasks'], num_subjects=loader_info['train']['num_subjects'])

    
    # select last GPU if multiple are available
    gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
    task_weights = compute_class_weights(train_loader, device=gpu)
    weights_dict = {'task': task_weights}

    config.data_shape = data_shape
    config.mid_channels = args.mid_channels

    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False, num_classes=loader_info['train']['num_tasks'])
    proj_out = backbone.model.proj_out if hasattr(backbone.model, 'proj_out') else None
    backbone.model.proj_out = nn.Identity()  # Ensure identity for feature extraction
    
    model = DisentangledEEGModel(backbone, config=config, phase_name='BB_FT', reconstruction_decoder=proj_out, classifier_type=args.classifier_type, skip_backbone=args.skip_backbone)
    model.to(gpu)

    backbone_freeze = config.backbone_freeze

    model.feature_extractor.freeze() if backbone_freeze else model.feature_extractor.unfreeze()
    
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
        
    stage1_trainer = DVAETrainer(model, loss_fn, optimizer, save_dir=args.save_dir, phase_name="BB_FT", device=gpu, xgb_classifier=args.xgb_classifier, tsne=args.tsne)
    stage1_trainer.train(train_loader, val_loader, test_loader,args.stage1_epochs, wandb_run=wandb if args.use_wandb else None)

    config_vars = vars(args).copy()
    
    torch.save(config_vars, Path(args.save_dir) / 'run_config.pt')  

    if args.use_wandb:
        wandb.finish()