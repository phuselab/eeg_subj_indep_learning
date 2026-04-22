import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any
from configs.config import ModelConfig
from einops.layers.torch import Rearrange
from utils.helper import FreezeUnfreeze, apply_euclidean_alignment
import torch.nn.functional as F


# 12k params
class VAEEncoderBody(nn.Module):
    def __init__(self, 
                 num_channels: int = 64, 
                 mid_channels: List[int] = [64, 32, 16, 8],
                 kernel_sizes: List[int] = [3, 3, 3],
                 pool_filter_size: int = 2,
                 use_skip: bool = True):
        super().__init__()

        self.use_skip = use_skip
        self.num_channels = num_channels
        
        self.encoders = nn.ModuleList()
        for i in range(len(mid_channels) - 1):
            in_ch = mid_channels[i]
            out_ch = mid_channels[i+1]
            k = kernel_sizes[i]
            self.encoders.append(self.block(in_ch, out_ch, kernel_size=k))
        self.pool = nn.MaxPool1d(pool_filter_size) # half temporal dimension at every down stage 

        # Final temporal bottleneck
        self.temporal_bottleneck = self.block(mid_channels[-1], mid_channels[-1], kernel_size=3)


    def block(self, in_ch, out_ch, kernel_size=3):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.ELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.ELU(),
        )
    

    def forward(self, x):
        # Input: (B, C, NP, T)
        B, C, NP, T = x.shape

        # Merge B and NP to apply 1D convolutions on time dimension across channels
        x = x.permute(0, 2, 1, 3).reshape(B * NP, C, T)
        
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)
            if self.use_skip:
                skips.append(x) # Save pre-pooling features for skip connections

        # --- 3. TEMPORAL BOTTLENECK ---
        out = self.temporal_bottleneck(x)  # B*NP, 64 // 2**num_pool, temporal_dim//2**num_pool
        c_red, t_red = out.shape[1], out.shape[2] 

        # --- 4. RESHAPE TO ORIGINAL STRUCTURE ---
        # Restore to (B, C_reduce, NP, T)
        out = out.view(B, NP, c_red, t_red).permute(0, 2, 1, 3) # B, C_RED, NP, T_RED

        
        # Reshape all skip connections similarly
        formatted_skips = []
        if self.use_skip:
            for skip in skips:
                c_red, t_red = skip.shape[1], skip.shape[2]
                s = skip.view(B, NP, c_red, t_red).permute(0, 2, 1, 3)
                formatted_skips.append(s)
            
            formatted_skips = formatted_skips[:-1] # Remove the skip from the last layer which is the input to the variational head

        # Return bottleneck features and reversed skips (deepest to shallowest)
        return out, formatted_skips[::-1] # out # B, C_RED, NP, T_RED
    

# 800k params 
class LatentClsHead(nn.Module):
    def __init__(self, 
                 num_of_classes: int, 
                 dropout: float = 0.1, 
                 mid_channels: List = [64,32,16,8],
                 patch_size: int = 200,
                 pool_filter_size: int = 2):
            
        super().__init__()
        
        self.patch_size = patch_size
        self.t_bottle_dim = self.patch_size // (pool_filter_size**(len(mid_channels)-1)) # Calculate the temporal dimension after all pooling layers, where mid_channels is the feature dimension of each encoder layer
        self.c_bottle_dim = mid_channels[-1] # The channel dimension at the bottleneck

        # Ora l'input lineare è leggermente più grande, ma preserva il tempo
        linear_input_dim = self.c_bottle_dim * self.t_bottle_dim
        mid_dim = int(self.patch_size * 1.3)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(linear_input_dim, mid_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(mid_dim, num_of_classes)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C_red, NP, T_RED)
        B, C_red, NP, T_red = features.shape
        
        x = features.permute(0, 2, 1, 3).contiguous() # (B, NP, C_red, T_red)
        x = x.reshape(B * NP, C_red * T_red) # (B*NP, C_red*T_red) with flattening

        x = self.linear(x) # (B*NP, mid_dim)

        x = x.reshape(B, NP, -1).permute(0, 2, 1) # (B, mid_dim, NP)

        x = self.pool(x) # B, mid_dim, 1
        x = x.squeeze(-1) # B, mid_dim
        
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
    
class VAEEncoderVarHeads(nn.Module, FreezeUnfreeze):
    def __init__(self, mid_channels: List = [64,32,16,8], patch_size: int = 200, pool_filter_size: int = 2):
        # we stay at 800 for the latent dim to later reshape back to (B, C, num_patches, BOTTLE_F_DIM, 25) and then upsample back to (B, 64, 4, 200) through the generator
        super().__init__()
        # VAE heads

        self.prev_dim = (mid_channels[-1] * (patch_size // (pool_filter_size**(len(mid_channels)-1)))) # Calculate the temporal dimension after all pooling layers, where nf_list is the feature dimension of each encoder layer
        self.fc_mu = nn.Linear(self.prev_dim, self.prev_dim) 
        self.fc_logvar = nn.Linear(self.prev_dim, self.prev_dim)

    
    def forward(self, x: torch.Tensor) -> dict:
        B, C_RED, NP, T_RED = x.shape # (B, C_RED, NP, T_RED)
        x = x.permute(0, 2, 1, 3).contiguous() # (B, NP, C_RED, T_RED)
        x = x.reshape(B * NP, C_RED * T_RED) # (B*NP, C_RED*T_RED) with flattening
        
        mu = self.fc_mu(x)
        mu = torch.clamp(mu, min=-10, max=10)

        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # reshape z back to (B, C, num_patches, num_features, temporal_dimension) to feed into the generator
        z = z.reshape(B, NP, C_RED, T_RED) # (B, NP, C_RED, T_RED)
        z = z.permute(0, 2, 1, 3).contiguous() # (B, C_RED, NP, T_RED)
 
        mu = mu.view(B, NP, C_RED, T_RED).permute(0, 2, 1, 3).contiguous() # (B, C_RED, NP, T_RED)
        logvar = logvar.view(B, NP, C_RED, T_RED).permute(0, 2, 1, 3).contiguous() # (B, C_RED, NP, T_RED)

        return {'z': z, 'mu': mu, 'logvar': logvar} # (B, C_RED, NP, T_RED)
        
# module to combine and project the concatenated latent codes back to a summary representation
class Combiner(nn.Module, FreezeUnfreeze):
    # !TODO FIX!
    def __init__(self, channels_list: List[int]):
        super().__init__()
        self.total_latent_dim = int(sum(channels_list)) # sum BOTTLE_F_DIM+BOTTLE_F_DIM+BOTTLE_F_DIM brororororooro
        self.combiner = nn.Sequential(
            Rearrange('num t -> num t'), # put feature dimension at the end to operate on them
            nn.Linear(self.total_latent_dim, int(self.total_latent_dim // len(channels_list))), # project to same dimension to allow for skip connection
            Rearrange('num t  -> num t')
        )


# Assumendo che Combiner e BOTTLE_F_DIM siano già importati

class Generator(nn.Module):
    def __init__(self,
                 mid_channels: List[int] = [64, 32, 16, 8], # Must mirror encoder's nf_list backwards
                 num_encoders: int = 3, # Number of encoders to combine (e.g., 3 for 3 factors)
                 patch_size: int = 200, 
                 learnable_combiner: bool = False,
                 proj_out: Optional[nn.Module] = None,
                 pool_filter_size: int = 2,
                 use_skip: bool = True) -> None:
        super().__init__()   
        
        self.patch_size = patch_size
        self.learnable_combiner = learnable_combiner
        self.rev_mid_channels = mid_channels[::-1] # reverse the mid_channels list
        
        self.num_encoders = num_encoders

        self.use_skip = use_skip
   
       
        # Mul handles the concatenation expansion if no learnable combiner is used
        mul = 1 if learnable_combiner else num_encoders
        
        # --- 1. DYNAMIC COMBINERS ---
        self.bottleneck_combiner = Combiner([self.rev_mid_channels[0]] * self.num_encoders) if learnable_combiner else None
        self.skip_combiners = nn.ModuleList()
        
        # --- 2. DYNAMIC UPSAMPLING DECODER ---
        self.up_layers = nn.ModuleList()
        
        # [8, 16, 32, 64]
        for i in range(len(self.rev_mid_channels) - 1):
            in_ch = self.rev_mid_channels[i] * mul # 24 -> 48
            out_ch = self.rev_mid_channels[i+1] * mul # 48
            if i == len(self.rev_mid_channels) - 2 and not learnable_combiner: # Last layer should output 1 channel for the final reconstruction
                out_ch = self.rev_mid_channels[-1]
            
            # Upsample temporal dimension by 2
            self.up_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=pool_filter_size, mode='linear'), # Upsample temporal dimension by 2
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # channel combination
                )
            )
            
            # SKIP connections combiners (created for all layers except the last one)
            if learnable_combiner and i < (len(self.rev_mid_channels) - 2):
                base_dim = self.rev_mid_channels[i+1] # as there is the bottleneck_combiner
                self.skip_combiners.append(Combiner([base_dim] * num_encoders)) # 48 -> 16
        
        self.proj_out = proj_out if proj_out is not None else nn.Identity()

    def forward(self, latent_codes: Dict[str, torch.Tensor], encoder_body_residuals: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        sorted_keys = sorted(latent_codes.keys())
        
        # --- 1. LATENT FUSION ---
        # Cat along C_RED dim: (B, C_RED, NP, T_RED)
        z = torch.cat([latent_codes[k] for k in sorted_keys], dim=1)
        B, C_RED_3, NP, T_RED = z.shape

        z = z.permute(0, 2, 1, 3).contiguous() # (B, NP, C_RED_3, T_RED)
        
        # Merge into 1D-compatible shape: (B*C_red*NP, T)
        x = z.view(B * NP, C_RED_3, T_RED)
        
        if self.learnable_combiner:
            x = self.bottleneck_combiner(x)

        # --- 2. UPSAMPLING & DYNAMIC SKIP CONNECTIONS ---
        if self.use_skip:
            num_skips = len(encoder_body_residuals[sorted_keys[0]])
        else:
            num_skips = 0
        
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x) # Upsample temporal dimension
            
            # Add skip connection if available for this depth level
            if i < num_skips:
                # Gather skips from all encoders for the i-th level
                skips_at_i = [encoder_body_residuals[k][i] for k in sorted_keys]
                
                # Cat along feature dim (dim=3 in original shape)
                cat_skip = torch.cat(skips_at_i, dim=1) # (B, C_RED*3, NP, T_RED) after concatenation along feature dimension
                cat_skip = cat_skip.permute(0, 2, 1, 3).contiguous() # (B, NP, C_red*3, T) to (B*NP, C_red*3, T)
                # Reshape to match x: (B*NP, C_red*3, T_skip)
                merged_skip = cat_skip.view(B * NP, -1, cat_skip.shape[-1]) # (B*NP, C_red*3, T_skip)
                
                if self.learnable_combiner:
                    merged_skip = self.skip_combiners[i](merged_skip) # (B*NP, C_red, T_skip)
                    
                x = x + merged_skip # Add residual
        
        # B * NP, 64, 200

        x = x.view(B, NP, -1, x.shape[-1]).permute(0, 2, 1, 3).contiguous() # (B, C_red, NP, T)

        x_hat = self.proj_out(x)
        x_hat = x_hat.reshape(x_hat.size(0), x_hat.size(1), -1) # Reshape to (B, num_channels, NP*Patch_size) for final output

        return x_hat


class Discriminator(nn.Module, FreezeUnfreeze):
    def __init__(self, num_channels: int = 64, patch_size: int = 200, num_patches: int = 4, dropout: float = 0.1):
        super().__init__()
        
        mid_channels = num_channels // 2
        patch_flat_dim = mid_channels * (patch_size * num_patches // 2) # After one Conv1d with stride 2, temporal dimension is halved, and channels are reduced to mid_channels
        critic_hidden_dim = patch_size // 2
        
        # Evaluates patches individually
        self.signal_evaluator = nn.Sequential(
            nn.Conv1d(num_channels, mid_channels, kernel_size=3, padding=1, stride=2), # (B, mid_channels, patch_size//2)
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(patch_flat_dim, critic_hidden_dim), # No more 128
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, 1) #! no sigmoid for WGAN, output is a real-valued score for each patch
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Reshape to process patches independently
        # Get "realness" score per patch
        patch_scores = self.signal_evaluator(x) # (B*N, 1)
        
        # Average the scores across all patches for each sample
        return patch_scores


class CBraModClassifier(nn.Module, FreezeUnfreeze):
    def __init__(self, num_channels: int = 64, num_patches: int = 4, patch_size: int = 200, config: Optional[ModelConfig] = None):
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.alignment_type = config.alignment_type

        self.flattened_input_dim = self.num_channels * self.num_patches * self.patch_size
        
        print(f'\n\n\n\n\n\nUsing classifier with alignment type: {self.alignment_type}\n\n\n\n\n\n')
        if self.alignment_type == None or self.alignment_type == 'euclidean':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.flattened_input_dim, self.num_patches * self.patch_size),
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout
                nn.Linear(self.num_patches * self.patch_size, self.patch_size),
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout
                nn.Linear(self.patch_size, config.encoders['task'].num_classes),
            )
        elif self.alignment_type == 'latentAlignment2D':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.BatchNorm1d(self.flattened_input_dim, affine=False, track_running_stats=False),

                nn.Linear(self.flattened_input_dim, self.num_patches * self.patch_size),
                nn.BatchNorm1d(self.num_patches * self.patch_size, affine=True, track_running_stats=False),
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout

                nn.Linear(self.num_patches * self.patch_size, self.patch_size),
                nn.BatchNorm1d(self.patch_size, affine=True, track_running_stats=False),
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout

                nn.Linear(self.patch_size, config.encoders['task'].num_classes),
            )
        elif self.alignment_type == 'batch_norm' or self.alignment_type == 'adaptive_batch_norm':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.BatchNorm1d(self.flattened_input_dim, affine=False), # Normalizzazione sull'input (come da paper [cite: 227])
                
                nn.Linear(self.flattened_input_dim, self.num_patches * self.patch_size),
                nn.BatchNorm1d(self.num_patches * self.patch_size, affine=False), # Normalizzazione dopo primo layer lineare
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout
                
                nn.Linear(self.num_patches * self.patch_size, self.patch_size),
                nn.BatchNorm1d(self.patch_size, affine=False), # Normalizzazione dopo secondo layer lineare
                nn.ELU(),
                nn.Dropout(0.1), # default CBraMod dropout

                nn.Linear(self.patch_size, config.encoders['task'].num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:                
        logits = self.classifier(x) # (B, num_classes)
        return logits
            


class DisentangledEEGModel(nn.Module, FreezeUnfreeze):
    """
    Modular disentangled representation learning model.
    This model can be used in two stages:
    1. 'pretrain': Trains a feature extractor and an encoder body with a simple classifier head.
    2. 'full': Uses the pretrained and frozen feature extractor and encoder body as a backbone for a full disentangled VAE model.
    """
    
    def __init__(self, feature_extractor: Optional[nn.Module], config: Any, 
                 classifier_type: str = 'diva_classifier', phase_name: str = 'BB_FT', reconstruction_decoder: Optional[nn.Module] = None, skip_backbone: bool = False):
        super().__init__()
        
        self.config = config
        self.feature_extractor = feature_extractor
        self.classifier_type = classifier_type
        self.alignment_type = self.config.alignment_type
        self.skip_backbone = skip_backbone

        self.num_channels = config.num_channels if hasattr(config, 'num_channels') else 64
        self.num_patches = config.num_patches if hasattr(config, 'num_patches') else 4
        self.patch_size = config.patch_size if hasattr(config, 'patch_size') else 200
        self.use_skip = config.use_skip

        self.pool_filter_size = config.pool_filter_size if hasattr(config, 'pool_filter_size') else 2
        
        self.phase_name = phase_name
        
        dropout = config.dropout if hasattr(config, 'dropout') else 0.1
        
        self.encoder_bodies = nn.ModuleDict()
        self.pretrain_classifier_head = nn.ModuleDict()
        self.variational_heads = nn.ModuleDict()
        self.var_classifier_heads = nn.ModuleDict()
        #self.decoders = nn.ModuleDict()
        
        self.encoder_names = []
        if hasattr(config, 'encoders'):
            self.encoder_names = sorted([name for name, enc in config.encoders.items() if enc.enabled])

        
        for name in self.encoder_names:
            enc_config = config.encoders[name]
            self.encoder_bodies[name] = VAEEncoderBody(
                num_channels=self.num_channels,
                mid_channels=config.mid_channels if hasattr(config, 'mid_channels') else [64, 32, 16, 8],
                kernel_sizes=config.kernel_sizes if hasattr(config, 'kernel_sizes') else [3, 3, 3],
                pool_filter_size=self.pool_filter_size,
                use_skip=self.use_skip
            )
            if 'noise' not in name:
                # the patch size is 25 because it is the result of the encoder body which reduces the temporal dimension from 200 to 25 through pooling (2**3 = 8 reduction)
                self.pretrain_classifier_head[name] = LatentClsHead(num_of_classes=enc_config.num_classes, dropout=dropout,
                                                                    mid_channels=config.mid_channels if hasattr(config, 'mid_channels') else [64, 32, 16, 8], patch_size=self.patch_size,
                                                                    pool_filter_size=self.pool_filter_size)
            # BOTTLE_F_DIM and 25 are the bottleneck num_features and temporal dimension
            self.variational_heads[name] = VAEEncoderVarHeads(mid_channels=config.mid_channels if hasattr(config, 'mid_channels') else [64, 32, 16, 8], patch_size=self.patch_size,
                                                              pool_filter_size=self.pool_filter_size)
            if 'noise' not in name:
                self.var_classifier_heads[name] = LatentClsHead(num_of_classes=enc_config.num_classes, dropout=dropout,
                                                                mid_channels=config.mid_channels if hasattr(config, 'mid_channels') else [64, 32, 16, 8], patch_size=self.patch_size,
                                                                pool_filter_size=self.pool_filter_size)


        self.generator = Generator(
            num_encoders=len(config.encoders.keys()), # MAGIC NUMBER BOTTLE_F_DIM because it's the output feature dimension of the encoder bodies, we want to keep it the same for the generator to mirror the encoder's structure
            mid_channels=config.mid_channels if hasattr(config, 'mid_channels') else [64, 32, 16, 8],
            patch_size=self.patch_size,
            learnable_combiner=False,
            proj_out=reconstruction_decoder,
            pool_filter_size=self.pool_filter_size,
            use_skip=self.use_skip
        )
        self.discriminator = Discriminator(
            num_channels=self.num_channels,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            dropout=dropout
        )

        # Default CBraMod classifier head as a baseline for ablation studies, to be used with the backbone features without disentanglement
        if self.classifier_type == 'cbramod_classifier':
            self.baseline_classifier =  CBraModClassifier(num_channels=self.num_channels, num_patches=self.num_patches, patch_size=self.patch_size, config=config)
        elif self.classifier_type == 'labram_classifier':
            self.baseline_classifier = nn.Sequential(
                nn.Linear(200, config.encoders['task'].num_classes)
            )
        # elif self.classifier_type == 'labram_classifier':
        #     self.baseline_classifier = nn.Identity()
        elif self.classifier_type == 'eegpt_classifier':
            print("Using EEGPT classifier, skipping baseline classifier head because it classifies directly")
            self.baseline_classifier = nn.Identity() # EEGPT will be used as the feature extractor, so we can skip the baseline classifier head

    def extract_features(self, x: torch.Tensor, skip_backbone: bool = False) -> torch.Tensor:
        """Extracts features using the feature extractor."""
        if self.feature_extractor is None:
            return x
        
        freeze = hasattr(self.config, 'freeze_feature_extractor') and self.config.freeze_feature_extractor
        if freeze:
            with torch.no_grad():
                return self.feature_extractor(x, skip_backbone=skip_backbone)
        else:
            return self.feature_extractor(x, skip_backbone=skip_backbone)

    def forward(self, x: torch.Tensor, extract_features:bool = True, do_reconstruction: bool = True, do_classification: bool = True) -> Dict[str, Any]:
        """
        Forward pass for the model. Returns a dictionary containing:
        - features_dict: Dictionary of features from each encoder body.
        - logits_dict: Dictionary of logits from each classifier head.
        - encoded_dict: Dictionary of encoded latent variables from each variational head (if in 'full' mode).
        - reconstruction: Reconstructed input from the generator (if in 'full' mode).
        
        
        Args:
            x (torch.Tensor): Input tensor.
            mode (str): 'pretrain' or 'full'.
        """
        
        if self.alignment_type == 'euclidean':
            x = apply_euclidean_alignment(x)
        
        if extract_features:
            backbone_features = self.extract_features(x, self.skip_backbone) # (B, C, NP, T)
        else:
            backbone_features = x

        encoder_body_bottleneck = {}
        encoder_body_residuals = {}
        if self.classifier_type == 'diva_classifier':
            for name in self.encoder_names:
                body = self.encoder_bodies[name]
                encoder_body_bottleneck[name], encoder_body_residuals[name]  = body(backbone_features) # after encoder 


        logits_dict = {}
        if do_classification:
            if self.classifier_type == 'diva_classifier' or self.phase_name == 'DVAE': 
                for name in self.encoder_names:
                    if 'noise' not in name:
                        logits_dict[name] = self.pretrain_classifier_head[name](encoder_body_bottleneck[name])
            else:
                logits_dict['task'] = self.baseline_classifier(backbone_features)

        var_features_dict = None 
        reconstruction = None
        var_logits_dict = None

        
        # Second stage: Full disentanglement model
        if self.phase_name == 'DVAE':
            var_logits_dict = {}
            var_features_dict = {}
            for name in self.variational_heads:
                encoded_var = self.variational_heads[name](encoder_body_bottleneck[name])
                var_features_dict[name] = encoded_var  # Update features_dict to hold encoded outputs
                if do_classification and 'noise' not in name:
                    var_logits_dict[name] = self.var_classifier_heads[name](encoded_var['z'])

            if do_reconstruction:
                reconstruction = self.decode_dvae(var_features_dict, encoder_body_residuals)
 
                
            
        return {
            'backbone_features': backbone_features,
            'encoder_body_residuals': encoder_body_residuals,
            # 'encoder_body_bottleneck': encoder_body_bottleneck,
            'logits_dict': logits_dict,
            'var_features_dict': var_features_dict,
            'var_logits_dict': var_logits_dict,
            'eeg_reconstruction': reconstruction,
        }
        
    def set_phase(self, phase_name: str):
        self.phase_name = phase_name

    # def encode_dvae(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    #     """Encodes features using the VAE encoders.
    #         Ret"""
    #     encoded = {}
    #     for name in self.encoder_names:
    #         encoded[name] = self.encoder_bodies[name](features_dict[name])
    #     return encoded
        
    
    def encode_dvae_var_heads(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Encodes features using the VAE variational heads."""
        encoded = {}
        for name in self.encoder_names:
            encoded[name] = self.variational_heads[name](features_dict[name])
        return encoded
    
    def decode_dvae(self, encoded: Dict[str, Dict[str, torch.Tensor]], encoder_body_residuals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decodes latent codes to reconstruct features."""
        latent_codes = {name: encoded[name]['z'] for name in self.encoder_names}
        return self.generator(latent_codes, encoder_body_residuals)

    def count_parameters(self) -> int:
        total_params = sum(p.numel() for p in self.parameters())
        learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - learnable_params
        
        print(f"Total parameters: {total_params}")
        print(f"Learnable parameters: {learnable_params}")
        print(f"Frozen parameters: {frozen_params}")
        
        return total_params
    
    def print_list_parameters(self) -> None:
        for name, module in self.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            status = "Trainable" if trainable else "Frozen"
            print(f"{name}: {status}")
            logging.info(f"  Number of parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad)}")
        self.count_parameters()
