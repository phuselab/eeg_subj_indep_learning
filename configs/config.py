from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import yaml
from dacite import from_dict

@dataclass
class EncoderConfig:
    """Configuration for a single encoder."""
    name: str
    enabled: bool = True
    num_classes: Optional[int] = None  # None = no classifier

@dataclass
class LossConfig:
    """Configuration for all losses."""
    # Reconstruction losses
    self_reconstruction: bool = False
    self_reconstruction_weight: float = 0.01
    self_reconstruction_weight_mse: float = 0.01  
    
    # VAE losses
    kl_divergence: bool = False
    kl_weight: float = 0.001  # β parameter
    noise_kl_weight: float = 0.0001  # Separate weight for noise encoder
    
    # Classification losses
    classification: bool = False
    classification_weight: float = 50.0

    # Classification loss on the z vector after variational head
    var_classification: bool = False
    var_classification_weight: float = 10.0
    
    # Consistency losses
    self_cycle: bool = False
    self_cycle_weight: float = 0.5
    
    # Cross-subject losses
    cross_subject_intra_class: bool = False
    cross_subject_intra_class_weight: float = 0.3
    cross_subject_cross_class: bool = False
    cross_subject_cross_class_weight: float = 0.3
    
    # Cross-class cycle consistency
    cross_cross_cycle: bool = False
    cross_cross_cycle_weight: float = 0.1
    
    # Knowledge distillation
    knowledge_distillation: bool = False
    kd_weight: float = 0.5
    
    # Adversarial losses (WGAN-GP)
    adversarial: bool = False
    adversarial_weight: float = 0.01
    lambda_gp: float = 10.0

    
    adaptive_balancing: bool = False

@dataclass
class ModelConfig:
    """Complete model configuration."""
    # Feature extractor settings
    num_channels: int = 64
    time_samples: int = 2048
    patch_size: int = 200
    num_patches: int = time_samples // patch_size
    classifier_type: str = 'avgpooling_patch_reps'
    dropout: float = 0.1
    use_skip: bool = True
    backbone_freeze: bool = False


    mid_channels: List[int] = field(default_factory=lambda: [64, 32, 16, 8])  # Example channel sizes for feature extractor
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_filter_size: int = 2

    by_subject: bool = False  # Whether to use subject-wise batching in the dataloader
    by_subject_inference: bool = False  # Whether to use subject-wise batching during inference (alignment)
    alignment_type: Optional[str] = None  # Type of alignment to apply (e.g., 'euclideanAlignment', 'latentAlignment2d', 'adaptiveBatchNorm', 'batchNorm')
    
    
    # Encoders (can add/remove as needed)
    encoders: Dict[str, EncoderConfig] = field(default_factory=lambda: {
        'subject': EncoderConfig('subject', enabled=True, num_classes=10),
        'task': EncoderConfig('task', enabled=True, num_classes=4),
        'noise': EncoderConfig('noise', enabled=True, num_classes=None),
        'device': EncoderConfig('device', enabled=False, num_classes=3),
        'session': EncoderConfig('session', enabled=False, num_classes=5),
    })
    
    # Loss configuration
    loss_config_stage1: LossConfig = field(default_factory=LossConfig)
    loss_config_stage2: LossConfig = field(default_factory=LossConfig)
    




def load_model_config(yaml_path: str, data_shape: tuple, num_subjects: int = 0, num_tasks: int = 0) -> ModelConfig:
    """
    Load configuration from YAML and inject dynamic parameters (shape, subjects, tasks).
    """
    # 1. Read YAML file into a dictionary
    with open(yaml_path, 'r') as file:
        raw_config = yaml.safe_load(file)
        
    # 2. Convert dict yaml to config dataclass
    config = from_dict(data_class=ModelConfig, data=raw_config)
    
    # 3. override dynamic parameters (shape, num_subjects, num_tasks)
    config.num_channels = data_shape[0]
    config.time_samples = data_shape[1]
    config.num_patches = data_shape[1] // config.patch_size
    
    # 4. Insert number of classes for subject and task encoders if they exist in the config
    if 'task' in config.encoders:
        config.encoders['task'].num_classes = num_tasks
    if 'subject' in config.encoders:
        config.encoders['subject'].num_classes = num_subjects
        
    logging.info(f"Config loaded from {yaml_path}")
    return config

def update_tasks_subjects(config, num_tasks, num_subjects):
    if 'task' in config.encoders:
        config.encoders['task'].num_classes = num_tasks
    if 'subject' in config.encoders:
        config.encoders['subject'].num_classes = num_subjects
    return config



# def create_config_diva(args, num_subjects: int, num_tasks: int, 


#                         data_shape: tuple, mid_channels: List[int] = [64, 32, 16, 8]) -> ModelConfig:
#     """
#     Create configuration for different training phases.
#     """
#     if args.phase == 'stage1':
#         logging.info("PHASE 1: only task classification")
#     if args.phase == 'stage2':
#         logging.info("PHASE 2: Task + Subject Classification")
    
#     logging.info("PHASE 2: Task + Subject Disentanglement")
#     config = ModelConfig(
#         num_channels=data_shape[0],
#         time_samples=data_shape[1],
#         patch_size = 200,
#         num_patches = data_shape[1] // 200,
#         classifier_type = 'avgpooling_patch_reps',
#         dropout = 0.1,
#         mid_channels=mid_channels,
#         kernel_sizes=[3, 3, 3],
#         pool_filter_size=2,

#         encoders={
#             'task': EncoderConfig(
#                 'task', 
#                 enabled=True, 
#                 num_classes=num_tasks,
#             ),
#             'subject': EncoderConfig(
#                 'subject',
#                 enabled=True,
#                 num_classes=num_subjects,
#             ),
#             'noise': EncoderConfig(
#                 'noise',
#                 enabled=True,
#                 num_classes=None,
#             ),
#         },
        
#         loss_config_stage1=LossConfig(
#             self_reconstruction=False,
#             self_reconstruction_weight=0.01,
#             self_reconstruction_weight_mse=0.01,
#             kl_divergence=False,
#             kl_weight=0.001,
#             noise_kl_weight=0.0001,
#             classification=True,
#             classification_weight=1.0,
#             var_classification=False,
#             var_classification_weight=10.0,
#             self_cycle=False,
#             self_cycle_weight=0.1,
#             cross_subject_intra_class=False,
#             cross_subject_intra_class_weight=0.5,
#             cross_subject_cross_class=False,
#             knowledge_distillation=False,
#             adversarial=False,
#             adaptive_balancing=False,
#         ),
#         loss_config_stage2 = LossConfig(
#             self_reconstruction=True,
#             self_reconstruction_weight=0.5,
#             self_reconstruction_weight_mse=0.5,

#             kl_divergence=True,
#             kl_weight=0.00001,
#             noise_kl_weight=0.000001,

#             classification=True,
#             classification_weight=0.5,

#             var_classification=True,
#             var_classification_weight=1.0,

#             cross_subject_intra_class=True,
#             cross_subject_intra_class_weight=0.5,

#             self_cycle=True,
#             self_cycle_weight=0.05,

#             cross_subject_cross_class=True,
#             cross_subject_cross_class_weight=0.1,

#             cross_cross_cycle = True,
#             cross_cross_cycle_weight = 0.05,

#             knowledge_distillation=True,
#             kd_weight=0.1,

#             adversarial=True,
#             adversarial_weight=0.1,
#         )

#     )
#     return config
