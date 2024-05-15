import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from .simple_mae_abs import Block
from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class BertConfig(Serializable):
    # data params
    window_size: int = 128
    n_electrodes: int = 256
    mask_ratio: float = 0.75

    n_layers: int = 8
    dim: int = 512
    hidden_dim: int = 1024

    head_dim: int = 32
    n_heads: int = 16
    n_kv_heads: int = 16 # now it should be the same with n_heads.


class BrainBert(nn.Module):
    def __init__(self, config, vq_model):
        super().__init__()
        self.config = config
        self.tokenizer = vq_model
        codebook_size = vq_model.codebook_size
        
        self.MASK_ID = codebook_size
        self.mask_ratio = config.mask_ratio
        
        self.transformer = nn.ModuleDict(dict(
            emb = nn.Embedding(codebook_size + 1, config.dim),
            space_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            time_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.dim),
        ))
        self.to_bt_c = Rearrange('b t c d -> (b t) c d', c=config.n_electrodes, t=config.window_size, d=config.dim)
        self.to_bc_t = Rearrange('(b t) c d -> (b c) t d', c=config.n_electrodes, t=config.window_size, d=config.dim)
        self.to_b_t_c = Rearrange('(b c) t d -> b t c d', c=config.n_electrodes, t=config.window_size, d=config.dim )
        

        self.spatial_pe = nn.Parameter(torch.randn(1, 1, config.n_electrodes, config.dim))
        self.time_pe = nn.Parameter(torch.randn(1, config.window_size, 1, config.dim))

        self.linear = nn.Linear(config.dim, codebook_size, bias=True)
        # self.attn_mask = torch.ones(config.block_size+self.n_registers, config.block_size+self.n_registers).to(torch.bool)
        self.attn_mask = None
        print(config)
        print("Encoder: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_attn_mask_padded(self, x, pad_value=0):
        is_padded = (x == pad_value).all(dim=2)
        not_padded_registers = torch.zeros(x.size(0), self.n_registers, device=x.device).to(torch.bool)
        is_padded = torch.cat([not_padded_registers, is_padded], axis=1)
        attn_mask = ~is_padded.unsqueeze(1).repeat(1, is_padded.size(1), 1)
        return attn_mask
        
    def forward(self, x, attn_mask=None):
        """
        myo signals with shape: with shape [B, T, C]
        Apply spatial/tempoeral attention.
        """
        # attn_mask = self.attn_mask if attn_mask is None else attn_mask
        # attn_mask = attn_mask.to(self.device)

        b, t, c = x.size()
        indices_out, _ = self.tokenizer.get_quantize_vectors(x)
        indices_out = indices_out.to(torch.long)
        indices = indices_out.contiguous()

        if self.mask_ratio != 0.0:
            indices = self.add_mask(indices)

        tokens = self.transformer.emb(indices)
        tokens = tokens + self.spatial_pe + self.time_pe

        for space_block, time_block in zip(self.transformer.space_blocks, self.transformer.time_blocks):
            tokens_space = space_block(self.to_bt_c(tokens))
            tokens_time = time_block(self.to_bc_t(tokens_space))
            tokens = self.to_b_t_c(tokens_time)

        x = self.transformer.ln_f(tokens)

        if self.mask_ratio !=0.0:
            x = self.linear(x)
            loss = F.cross_entropy(x.view(-1, x.size(-1)), indices_out.reshape(-1))
            return loss, x
        else:
            return None, x
    
    def add_mask(self, x):
        num_to_mask = int(torch.numel(x) * self.mask_ratio)

        # Generate random indices to mask
        mask_indices = torch.randperm(torch.numel(x))[:num_to_mask]

        # Apply the mask directly to the reshaped version of indices
        x.view(-1)[mask_indices] = self.MASK_ID
        return x

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params