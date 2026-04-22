import torch
import torch.nn as nn
import time
import torch.nn.functional as F

#from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder #! original import, now relative
from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder


class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        self.seq_len = seq_len # added
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )
        # lazy projection: created on first forward if conv flattened dim != d_model
        self.to_d_model = None
        self._proj_initialized = False
        self._spec_initialized = False          # ! Lia added
        self.spectral_proj = None  # will lazy-init based on FFT bins # ! Lia added

    def forward(self, x, mask=None):
        #print("PatchEmbedding input shape:", x.shape)
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size) 
        # ! original
        #patch_emb = self.proj_in(mask_x)
        #patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)
        # ! original end 
        
        conv_out = self.proj_in(mask_x)  # (bz, out_ch, ch_num*patch_num, out_time)
        # reshape conv output into (bz, ch_num, patch_num, conv_flat_dim)
        conv_out = conv_out.permute(0, 2, 1, 3).contiguous()  # (bz, ch_num*patch_num, out_ch, out_time)
        conv_out = conv_out.view(bz, ch_num, patch_num, patch_size)  # last dim == conv_flat_dim

        
        # initialize projection lazily if needed (handles arbitrary patch_size)
        conv_flat_dim = conv_out.shape[-1]
        '''
        if not self._proj_initialized:
            if conv_flat_dim != self.d_model:
                self.to_d_model = nn.Linear(conv_flat_dim, self.d_model)
                # move proj to same device as conv_out
                self.to_d_model.to(conv_out.device)
            else:
                self.to_d_model = None
            self._proj_initialized = True
        '''
        if not self._proj_initialized:
            self.to_d_model = None if conv_flat_dim == self.d_model else nn.Linear(conv_flat_dim, self.d_model).to(conv_out.device)
            self._proj_initialized = True
        '''
        if self.to_d_model is not None:
            conv_flat = conv_out.view(-1, conv_flat_dim)  # (bz*ch*patch, conv_flat_dim)
            conv_proj = self.to_d_model(conv_flat).view(bz, ch_num, patch_num, self.d_model)
        else:
            conv_proj = conv_out  # already sized to d_model
        '''
        
        if self.to_d_model is not None:
            print(f'Projecting conv output from {conv_flat_dim} to {self.d_model}')
            conv_proj = self.to_d_model(conv_out.view(-1, conv_flat_dim)).view(bz, ch_num, patch_num, self.d_model)
        else:
            conv_proj = conv_out  # already d_model

        # ! projection logic end 
        #mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        # ---- spectral path (make it shape-agnostic) ----
        
        
        # Flatten patches for FFT: (bz*ch*patch, patch_size)
        flat = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        
        '''
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        '''
        
        spectral = torch.fft.rfft(flat, dim=-1, norm='forward')    # (bz*ch*patch, nfft)
        spectral = torch.abs(spectral)
        nfft = spectral.shape[-1]                                  # should be patch_size//2 + 1

        # lazy-init spectral projector
        if not self._spec_initialized:
            self.spectral_proj = nn.Linear(nfft, self.d_model).to(spectral.device)
            self._spec_initialized = True
            
        spectral = spectral.view(bz, ch_num, patch_num, nfft)
        spectral_emb = self.spectral_proj(spectral)                # (bz, ch_num, patch_num, d_model)

        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        # patch_emb = patch_emb + spectral_emb #! orig
        # fuse + positional encoding
        patch_emb = conv_proj + spectral_emb # new

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb




def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)