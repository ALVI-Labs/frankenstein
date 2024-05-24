import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

from .blocks import Block
from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class EncoderConfig(Serializable):
    # data params
    window_size: int = 32
    n_electrodes: int = 256
    patch_size: int = 4

    n_layers: int = 12
    dim: int = 256
    hidden_dim: int = 1024

    head_dim: int = 32
    n_heads: int = 16
    n_kv_heads: int = 16 # now it should be the same with n_heads.

@dataclass
class MAEConfig(Serializable):
    masking_ratio: float = 0.5

    # data params
    n_layers: int = 6
    dim: int = 256
    hidden_dim: int = 1024

    head_dim: int = 16
    n_heads: int = 8
    n_kv_heads: int = 8 # now it should be the same with n_heads.



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_patches_time = int(config.window_size / config.patch_size)
        self.block_size = int(n_patches_time * config.n_electrodes)

        self.transformer = nn.ModuleDict(dict(
            emb = nn.Linear(config.patch_size, config.dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.dim),
        ))

        self.date_embeddings = nn.Embedding(num_embeddings=25, embedding_dim=config.dim)
        self.reshape_to_patches = Rearrange('b (t p1) c -> b t c p1', p1=config.patch_size)

        self.spatial_pe = nn.Parameter(torch.randn(1, 1, config.n_electrodes, config.dim))
        self.time_pe = nn.Parameter(torch.randn(1, n_patches_time, 1, config.dim))

        
        print(config)
        print("Simple Encoder: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        
    def forward(self, x, targets=None, date_info=None):
        """
        myo signals with shape: with shape [B, T, C]
        """

        patches = self.reshape_to_patches(x)
        patches = self.transformer.emb(patches)


        patches = patches + self.time_pe + self.spatial_pe
        patches = rearrange(patches, 'b t c p -> b (t c) p')

        date_emb = self.date_embeddings(date_info)
        x = torch.cat([date_emb, patches], dim=1)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


    def get_masking_indices(self, masking_ratio, x):
        b, n_tokens, _ = x.shape

        num_masked = int(masking_ratio * n_tokens)
        rand_indices = torch.rand(b, n_tokens, device = x.device).argsort(dim=-1) # get idxs of random values.
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        masked_indices, _ = torch.sort(masked_indices, dim=1)
        unmasked_indices, _ = torch.sort(unmasked_indices, dim=1)      
        
        return masked_indices, unmasked_indices

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def rope_cache(self) -> torch.Tensor:
        # Just to use proper device.
        if self.precompute_rope_cash.device != self.device:
            self.precompute_rope_cash = self.precompute_rope_cash.to(device=self.device)
        return self.precompute_rope_cash
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params



class MAE(nn.Module):
    def __init__(self, encoder_config, mae_config):
        super().__init__()
        self.encoder_config = encoder_config
        
        self.encoder = Encoder(encoder_config)
        self.dim = mae_config.dim
        self.masking_ratio = mae_config.masking_ratio

        self.decoder = nn.ModuleDict(dict(
            emb = nn.Linear(encoder_config.dim, mae_config.dim), # connection between them.
            h = nn.ModuleList([Block(mae_config) for _ in range(mae_config.n_layers)]),
        ))

        self.mask_token = nn.Parameter(torch.randn(mae_config.dim))
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, self.encoder.block_size + 1, mae_config.dim))
        self.proj_to_signals = nn.Linear(mae_config.dim, encoder_config.patch_size)

        print("MAE: number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def get_masking_indices(self, masking_ratio, x):
        b, n_tokens, _ = x.shape

        num_masked = int(masking_ratio * n_tokens)
        rand_indices = torch.rand(b, n_tokens, device = x.device).argsort(dim=-1) # get idxs of random values.
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        masked_indices, _ = torch.sort(masked_indices, dim=1)
        unmasked_indices, _ = torch.sort(unmasked_indices, dim=1)      
        
        return masked_indices, unmasked_indices
    

    def forward(self, x, targets=None, date_info=None, return_preds=False):
        """
        Inputs: x with shape -> B, T, C
        """
        b, t, c = x.shape
        batch_range = torch.arange(b, device=x.device)[:, None]
        
        # Encoder
        patches = self.encoder.reshape_to_patches(x) # b t c p
        
        embds = self.encoder.transformer.emb(patches)
        embds = embds + self.encoder.time_pe + self.encoder.spatial_pe
        full_tokens = rearrange(embds, 'b t c d -> b (t c) d')

        # masking 
        masked_indices, unmasked_indices = self.get_masking_indices(self.masking_ratio, full_tokens) # [b, N]
        tokens = full_tokens[batch_range, unmasked_indices]

        # cat cls tokens
        date_emb = self.encoder.date_embeddings(date_info)
        x = torch.cat([date_emb, tokens], dim=1) # b, 1 + (t c)*0.25, 
        

        for block in self.encoder.transformer.h:
            x = block(x)
        x = self.encoder.transformer.ln_f(x)

        
        ### DECODER 
        unmasked_decoder_tokens = self.decoder.emb(x)

        cls_token = unmasked_decoder_tokens[:, :1]
        data_tokens = unmasked_decoder_tokens[:, 1:]

        decoder_tokens = torch.zeros(b, self.encoder.block_size, self.dim, device=x.device)
        decoder_tokens[batch_range, unmasked_indices] = data_tokens
        decoder_tokens[batch_range, masked_indices] = self.mask_token

        decoder_tokens = torch.cat([cls_token, decoder_tokens], axis=1)
        decoder_tokens = decoder_tokens + self.decoder_pos_emb

        
        for block in self.decoder.h:
            decoder_tokens = block(decoder_tokens)
        
        decoder_tokens = decoder_tokens[:, 1:]
       
        ## loss calculation
        pred_tokens = self.proj_to_signals(decoder_tokens)

        tokens_pred_masked = pred_tokens[batch_range, masked_indices]

        patches = rearrange(patches, 'b t c p -> b (t c) p')
        tokens_real_masked = patches[batch_range, masked_indices]

        loss = F.mse_loss(tokens_pred_masked, tokens_real_masked)
            
        if return_preds:
            binary_mask = torch.zeros_like(x, device=x.device, dtype=x.dtype) 
            binary_mask[batch_range, masked_indices] = 1

            reconstruction_signal = torch.zeros_like(x, device=x.device, dtype=x.dtype)
            reconstruction_signal[batch_range, masked_indices] = pred_tokens[batch_range, masked_indices]
            reconstruction_signal[batch_range, unmasked_indices] = x[batch_range, unmasked_indices]

            return loss, reconstruction_signal, binary_mask

        return loss, None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @property
    def device(self) -> torch.dtype:
        return next(self.parameters()).dtype


