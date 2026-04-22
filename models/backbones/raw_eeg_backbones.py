import torch
import torch.nn as nn
import logging

class LearnableRawProjector(nn.Module):
    """
    Learnable projector for raw EEG data (Mode A).
    Includes normalization and better architecture.
    """
    def __init__(self, n_chans: int, n_times: int, hidden_dim: int = 512,
                 target_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        
        input_dim = n_chans * n_times
        self.feature_dim = target_dim
        
        # Learnable normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Deeper projector with residual connection
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, target_dim)
        )
        
        # Initialize with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG
        Returns:
            features: (B, target_dim)
        """
        B, C, T = x.shape
        
        # Flatten and normalize
        x_flat = x.reshape(B, -1)  # (B, C*T)
        x_norm = self.input_norm(x_flat)
        
        # Project
        features = self.projector(x_norm)
        return features
    

class CNNRawProjector(nn.Module):
    """
    CNN Projector for raw EEG data.
    Replaces the flat MLP with convolutional layers to preserve structure.
    """
    def __init__(self, n_chans: int, n_times: int, hidden_conv_dim: int = 128,
                 target_dim: int = 200, dropout: float = 0.2):
        super().__init__()
        
        self.feature_dim = target_dim

        # --- 1. CNN BACKBONE (Instead of flattening and MLP projector) ---
        self.cnn_backbone = nn.Sequential(
            # Adaptation from n_chans to a richer set of feature maps
            nn.Conv1d(n_chans, 64, kernel_size=16, stride=4, padding='same'), 
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, hidden_conv_dim, kernel_size=8, stride=2, padding='same'),
            nn.BatchNorm1d(hidden_conv_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(dropout),
        )

        # 2. Determine Dynamic Output Dimension after CNN
        dummy_input = torch.randn(1, n_chans, n_times)
        with torch.no_grad():
            cnn_output_shape = self.cnn_backbone(dummy_input).shape
        flat_dim = cnn_output_shape[1] * cnn_output_shape[2] 
        
        logging.info(f"CNN Projector: Output flattened dimension (F_dim) = {flat_dim}")

        # 3. Final Projection Layer (From Flat_dim to Target_dim=200)
        self.final_projection = nn.Sequential(
            nn.Linear(flat_dim, hidden_conv_dim), # Small MLP transition
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_conv_dim, target_dim) # Projects to DVAE feature dimension
        )
        
        # Initialize with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG
        Returns:
            features: (B, target_dim)
        """
        # 1. CNN Feature Extraction: (B, C, T) -> (B, C_out, T_out)
        cnn_output = self.cnn_backbone(x) 
        
        # 2. Flatten and Project
        x_flat = cnn_output.view(cnn_output.size(0), -1) 
        
        # 3. Final Projection to target_dim
        features = self.final_projection(x_flat)
        return features
