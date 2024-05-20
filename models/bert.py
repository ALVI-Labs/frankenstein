import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

from .simple_mae_abs import Block
from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class BertConfig(Serializable):
    # data params
    window_size: int = 128
    n_electrodes: int = 256
    mask_ratio: float = 0.75
    tokenizer_downsample: int = 4

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
        
        self.mask_ratio = config.mask_ratio
        self.n_electrodes = config.n_electrodes
        self.tokenizer_downsample = config.tokenizer_downsample
        self.dim = config.dim
        
        self.tokenizer = vq_model
        codebook_size = vq_model.codebook_size
        
        self.MASK_ID = codebook_size
        self.pad_value = 0
        
        self.transformer = nn.ModuleDict(dict(
            emb = nn.Embedding(codebook_size + 1, config.dim),
            space_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            time_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.dim),
        ))

        # self.to_bc_t = Rearrange('(b t) c d -> (b c) t d', c=config.n_electrodes, d=config.dim)
        # self.to_b_t_c = Rearrange('(b c) t d -> b t c d', c=config.n_electrodes, d=config.dim)
        

        self.spatial_pe = nn.Parameter(torch.randn(1, 1, config.n_electrodes, config.dim))
        self.time_pe = nn.Parameter(torch.randn(1, config.window_size //self.tokenizer_downsample, 
                                                1, config.dim))

        self.linear = nn.Linear(config.dim, codebook_size, bias=True)
        # self.attn_mask = torch.ones(config.block_size+self.n_registers, config.block_size+self.n_registers).to(torch.bool)
        self.attn_mask = None
        print(config)
        print("Encoder: number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    # def get_attn_mask_padded(self, x, pad_value=0):
    #     is_padded = (x == pad_value).all(dim=2)
    #     attn_mask = ~is_padded.unsqueeze(1).repeat(1, is_padded.size(1), 1)
    #     return attn_mask

    
    def get_attn_mask_padded_registers(self, x, pad_value=0):
        is_padded = (x == pad_value).all(dim=2)
        not_padded_registers = torch.zeros(x.size(0), self.n_registers, device=x.device).to(torch.bool)
        is_padded = torch.cat([not_padded_registers, is_padded], axis=1)
        attn_mask = ~is_padded.unsqueeze(1).repeat(1, is_padded.size(1), 1)
        return attn_mask
        
    def forward(self, x, targets=None, date_info=None, return_indices=False):
        """
        myo signals with shape: with shape [B, T, C]
        Apply separable space/time attention.

        1. get pad binary mask, which reflects time intervals in which all neuran activity equal 0.
        create attention mask to not use these tokens in time attention.

        2. apply tokenizer which get indices from brain activity and downsample them

        3. randomly masking tokens and change embedding for MASK

        4. process all tokens 

        5. classification loss 
            - masked tokens.
            - nonpadded tokens.        
        """
        # attn_mask = self.attn_mask if attn_mask is None else attn_mask
        # attn_mask = attn_mask.to(self.device)

        b, t, c = x.size()

        # binary padded elements, downscale for tokenizers, get attn mask for time
        
        is_padded = (x == self.pad_value).all(dim=2)[:, ::self.tokenizer_downsample]
        time_attn_mask = ~is_padded.unsqueeze(1).repeat(1, is_padded.size(1), 1)
        time_attn_mask = torch.repeat_interleave(time_attn_mask, self.config.n_electrodes, 0)
        time_attn_mask = time_attn_mask.unsqueeze(1) # b, 1, T, T
                
        indices_out = self.tokenizer.get_indices(x)       
        indices = torch.clone(indices_out)
        
        if self.mask_ratio != 0.0:
            indices = self.add_mask(indices)

        # print('indices_out', torch.max(indices_out))
        # print('indices', torch.max(indices))
        
        tokens = self.transformer.emb(indices)
        tokens = tokens + self.spatial_pe + self.time_pe
        
        b, t, c, d = tokens.size()

        tokens = rearrange(tokens, 'b t c d -> (b t) c d', b=b, t=t, c=self.n_electrodes, d=self.dim)
        
        for space_block, time_block in zip(self.transformer.space_blocks, self.transformer.time_blocks):
            tokens = space_block(tokens)
            tokens = rearrange(tokens, '(b t) c d -> (b c) t d', b=b, t=t, c=self.n_electrodes, d=self.dim)
            tokens = time_block(tokens, attn_mask=time_attn_mask)
            tokens = rearrange(tokens, '(b c) t d -> (b t) c d', b=b, t=t, c=self.n_electrodes, d=self.dim)

        tokens = rearrange(tokens, '(b t) c d -> b t c d', b=b, t=t, c=self.n_electrodes, d=self.dim)
        y = self.transformer.ln_f(tokens)

        loss = None
        if self.mask_ratio !=0.0:
            y = self.linear(y)
            
            is_padded = is_padded.unsqueeze(-1).expand(-1, -1, self.config.n_electrodes)
            # indices_out[indices!=self.MASK_ID] = -100 # masked only loss calculation
            indices_out[is_padded] = -100
            
            loss = F.cross_entropy(y.view(-1, y.size(-1)), indices_out.to(torch.long).view(-1))    
        if return_indices:
            return loss, y, indices
        return loss, y
    
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