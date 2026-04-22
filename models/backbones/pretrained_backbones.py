import torch
import torch.nn as nn
from braindecode.models import Labram, EEGPT
from CBraMod_main.models.cbramod import CBraMod
from utils.helper import segment_to_patches # This will need to be adjusted based on the new structure
from utils.helper import FreezeUnfreeze, clean_ch_names
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import mne


class FeatureBackbone(nn.Module):
    """Unified interface: forward(x[B,C,T]) -> features[B,F]."""
    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError
    
class LabramBackbone(FeatureBackbone, FreezeUnfreeze):
    def __init__(self, n_chans: int, n_times: int, weights_path: str | None, patch_size: int, emb_size: int = 200, ch_names: list = None):
        super().__init__()
        
        # 1. Re-enable the Montage (Critically important for spatial tokens)
        chs_info = None
        if ch_names is not None:
            ch_names = clean_ch_names(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, on_missing='ignore')
            chs_info = info['chs']

        # 2. Initialize Model as Feature Extractor (n_outputs=0)
        self.model = Labram(
            n_chans=n_chans, 
            n_times=n_times, 
            n_outputs=0, 
            patch_size=patch_size, 
            embed_dim=emb_size, 
            chs_info=chs_info
        )

        # 3. Load and Adapt State Dict
        model_path = hf_hub_download(repo_id="braindecode/labram-pretrained", filename="model.safetensors")
        state_dict = load_file(model_path)

        # Fix Temporal Mismatch: Slice the weights
        # Your model expects 5 tokens (4 patches + 1 CLS); checkpoint has 16.
        if "temporal_embedding" in state_dict:
            expected_tokens = self.model.temporal_embedding.shape[1] # e.g., 5
            state_dict["temporal_embedding"] = state_dict["temporal_embedding"][:, :expected_tokens, :]
            print(f"Adapted temporal_embedding to {expected_tokens} tokens.")

        # 4. Load with strict=False
        # This ignores the 'final_layer' keys which are missing in the checkpoint 
        # but exist in your model because it's initialized for n_outputs.
        self.model.load_state_dict(state_dict, strict=False)
        
        self.feature_dim = emb_size

    def forward(self, x: torch.Tensor, skip_backbone=False) -> torch.Tensor:
        return self.model.forward(x)

class EEGPTBackbone(FeatureBackbone, FreezeUnfreeze):
    def __init__(self, n_chans: int, n_times: int, num_classes: int = 4):
        super().__init__()
        self.model = EEGPT(n_chans=n_chans, n_times=n_times, n_outputs=num_classes)
        model_path = hf_hub_download(repo_id="braindecode/eegpt-pretrained", filename="model.safetensors")
        state_dict = load_file(model_path)
        del state_dict['chans_id']
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor, skip_backbone=False) -> torch.Tensor:
        output = self.model.forward(x)
        return output


class CBraModBackbone(FeatureBackbone, FreezeUnfreeze):
    def __init__(self, in_dim: int, time_samples: int, patch_size: int, weights_path: str, emb_dim: int = 200, use_identity_for_reconstruction: bool = True):
        super().__init__()
        #from CBraMod_main.models.cbramod import CBraMod
        from CBraMod_main.models.cbramod_original import CBraMod 
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            print(f"STFTLoss: Detected {num_devices} CUDA device(s). Using device: {torch.cuda.get_device_name(0)}")
        # select last GPU if multiple are available
        gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(gpu)
        seq_len = time_samples // patch_size
        self.seq_len = seq_len
        print(f"CBraModBackbone: Using in_dim={in_dim}, time_samples={time_samples}, patch_size={patch_size}, seq_len={seq_len}")
        self.model = CBraMod(in_dim=in_dim, out_dim=emb_dim, d_model=emb_dim, seq_len=seq_len).to(self.device)
        
        weights = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if any('backbone.' in k for k in weights.keys()):
            weights = {k.replace('backbone.', ''): v for k, v in weights.items()}
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier.')}
            
        self.model.load_state_dict(weights, strict=True)
        if use_identity_for_reconstruction:
            self.model.proj_out = nn.Identity()
        self.feature_dim = emb_dim

    def forward(self, x: torch.Tensor, segment_to_patches_enabled=True, skip_backbone=False) -> torch.Tensor:
        T = x.shape[-1]
        seq_len = self.seq_len
        patch_size = T // seq_len
        if segment_to_patches_enabled:
            patched = segment_to_patches(x, patch_size).to(self.device)
        else:
            patched = x.to(self.device)
        if skip_backbone:
            return patched
        out = self.model.forward(patched)
        
        return out
