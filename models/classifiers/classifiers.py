import torch
import torch.nn as nn
import logging
from typing import List, Optional, Any, Dict
from einops.layers.torch import Rearrange

# Local imports will need adjustment
from configs.config import ModelConfig


class ExternalClassifierHead(nn.Module):
    def __init__(self, classifier_type: str, feature_dim: int, num_of_classes: int, dropout: float = 0.1):
        super().__init__()
        if classifier_type == 'avgpooling_patch_reps':
            self.head = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64 * 4 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, num_of_classes),
            )
        else:
            raise NotImplementedError(f"Classifier type {classifier_type} not implemented.")

    def forward(self, x):
        return self.head(x)



class SimpleFeaturesClassifier(nn.Module):
    """Modular disentangled representation learning model."""
    
    def __init__(self, feature_extractor: Optional[nn.Module], feature_dim: int, 
                 config: ModelConfig, num_tasks: int = 4, num_subjects: int = 10, train_backbone: bool = True):

        super().__init__()
        
        self.config = config
        self.feature_extractor = feature_extractor
        self.train_backbone = train_backbone

        encoders_of_interest = self.config.encoders.keys()
        classifier_dict = {}

        for name in encoders_of_interest:
            
            if name not in ['noise']:
                classifier_dict[name] = ExternalClassifierHead(
                    classifier_type="avgpooling_patch_reps",
                    feature_dim=feature_dim,
                    num_of_classes=self.config.encoders[name].num_classes
                )
                # classifier_dict[name] = nn.Sequential(
                #     Rearrange('b c s d -> b (c s d)'),
                #     nn.Linear(64 * 4 * 200, 4 * 200),
                #     nn.ELU(),
                #     nn.Dropout(0.1),
                #     nn.Linear(4 * 200, 200),
                #     nn.ELU(),
                #     nn.Dropout(0.1),
                #     nn.Linear(200, self.config.encoders[name].num_classes),
                # )
        
        #self.feature_extractor.proj_out = nn.Identity() # unnecessary,(checked) as it is already there 
        self.classifier_dict = nn.ModuleDict(classifier_dict)
        #self.set_feature_pooling(False) # 
        # for name, classifier in self.classifier_dict.items():
        #     for m in classifier.modules():
        #         if isinstance(m, nn.Linear):
        #             # Forza i pesi a essere molto piccoli per compensare input grandi
        #             nn.init.xavier_uniform_(m.weight, gain=0.01) 
        #             # Forza il bias a 0 per garantire probabilità uniformi all'inizio
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        

    def extract_features(self, x: torch.Tensor, cbraLoader=0) -> torch.Tensor:
        if self.feature_extractor is None:
            return x
        if self.config.freeze_feature_extractor:
            with torch.no_grad():
                return self.feature_extractor(x, cbraLoader=cbraLoader)
        return self.feature_extractor(x, cbraLoader=cbraLoader)
    
    # def set_feature_pooling(self, enable_pooling: bool):
    #     if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'pool_output'):
    #         self.feature_extractor.pool_output = enable_pooling # TODO pool_output is not used anymore, we can delete it
    #         logging.info(f"Feature Extractor output pooling set to: {enable_pooling}")
    #     else:
    #         logging.warning("Feature extractor or pool_output attribute not found.")
            
    def set_required_grad_for_classifier(self, requires_grad: bool = True):
        for classifier in self.classifier_dict.values():
            for param in classifier.parameters():
                param.requires_grad = requires_grad
    
    def set_required_grad_for_backbone(self, requires_grad: bool = True):
        self.train_backbone = requires_grad
        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grad
    
    
    def forward(self, x: torch.Tensor, cbraLoader=0) -> Dict[str, Any]:
        features = self.extract_features(x, cbraLoader=cbraLoader)
        logits_dict = {}
        for name, classifier in self.classifier_dict.items():
            logits = classifier(features)
            logits_dict[name] = {'logits': logits}
        return logits_dict



class MLPClassifier(nn.Module):
    """Simple MLP classifier for features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        layers.extend([Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()])
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)