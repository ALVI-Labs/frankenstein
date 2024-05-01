import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from typing import Optional

from dataclasses import dataclass
from simple_parsing.helpers import Serializable
import time
import numpy as np
## Functions

## Default Configs
@dataclass
class MAEConfig(Serializable):
    # data params
    window_size: int = 1024
    n_electrodes: int = 256
    patch_size: int = 32

    # encoder
    dim: int = 256
    n_layers: int = 8
    head_dim: int = 32
    hidden_dim: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 8 # now it should be the same with n_heads.
    rope_theta: int = 10000

    # decoder 
    n_dec_layers: Optional[int] = 4
    decoder_dim: Optional[int] = 256

@dataclass
class Config(Serializable):
    encoder: MAEConfig

    # perciever
    n_output_tokens: int = 32
    output_dim: int = 1024
    
    dim: int = 256 # should be the same with encoder.
    n_layers: int = 2
    head_dim: int = 16
    hidden_dim: int = 512
    n_heads: int = 4
    n_kv_heads: int = 4
    rope_theta: int = 10_000


def build_complex_rope_cache(dim: int, seq_len: int, theta: float) -> torch.Tensor:
    """
    Compute cache for RoPE and store on device with complex dtype. 
    It speeds up computation.
    Return: [T, dim//2]
    """
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)  
    freqs = torch.outer(t, freqs).float()
    cache = torch.polar(torch.ones_like(freqs), freqs) 
    cache.requires_grad = False
    return cache

def apply_rope(x: torch.Tensor, rope: torch.Tensor):
    """
    Now, we do not cut rope cache.  You have to cut outside.
    x - [b, t, n_h, dim] 
    rope(freqs_cis):  [T, dim//2] or [B, T, dim//2]
    """
    T = x.size(1)
    len_rope = len(rope.shape)
    
    if len_rope == 2:
        rope = rope[-T:]
    else:
        rope = rope[:, -T:]

    rope = rope.unsqueeze(-2) # [B, T, 1, dim//2] or [T, 1, dim//2]

    # b, t, n_h, dim - > (b, t, n_h, dim/2, 2)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))    

    # (b, t, n_h, dim/2, 2) * t, 1, dim/2
    x_out = torch.view_as_real(x_ * rope).flatten(3)
    return x_out.type_as(x)

def build_advanced_causal_mask(block_size, tok_per_time):
    """
    Return mask in bool dtype, where
        True - include in attention 
        False - don't include.   
    """

    mask = torch.ones(block_size, block_size)
    mask = torch.tril(mask)

    S = torch.ones(tok_per_time, tok_per_time)

    for i in range(0, block_size, tok_per_time):
        lp, rp = i, i + tok_per_time
        mask[lp:rp, lp:rp] = S
    
    causal_mask = mask
    causal_mask = causal_mask.to(torch.bool)
    return causal_mask

## Blocks.

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module): 
    """
    Simple Multi head attention with einops and F.scaled_dot_product_attention.
    """
    def __init__(self, config, is_causal=True):

        super().__init__()

        assert config.n_heads == config.n_kv_heads, "n_heads should be equal n_kv_heads"

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads # for simplicity we use vanilla transformer.
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.qw = nn.Linear(config.dim, config.head_dim * config.n_heads, bias=False)
        self.kw = nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=False)
        self.vw = nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=False)

        self.project = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)

    def forward(self, x, attn_mask, rope, kv_cache=None):
        B, T, C = x.size() # b, t, c*h        
        q, k, v = self.qw(x), self.kw(x), self.vw(x) 

        # split by n_heads.
        q = rearrange(q, 'b t (nh c) -> b t nh c', b=B, t=T, nh=self.n_heads, c=self.head_dim)
        k = rearrange(k, 'b t (nh c) -> b t nh c', b=B, t=T, nh=self.n_heads, c=self.head_dim)
        v = rearrange(v, 'b t (nh c) -> b t nh c', b=B, t=T, nh=self.n_heads, c=self.head_dim)

        if rope is not None:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)

        if attn_mask is not None:
            t_q, t_k = q.size(1), k.size(1)
            attn_mask = attn_mask[..., -t_q:, -t_k:]

        q = q.transpose(1, 2)  # (B, nh, T, c)
        k = k.transpose(1, 2)  # (B, nh, T, c)
        v = v.transpose(1, 2)  # (B, nh, T, c)
        
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  

        res = rearrange(res, 'b nh t c -> b t (nh c)', b=B, t=T, nh=self.n_heads, c=self.head_dim)
        res = self.project(res)

        return res
    
class CausalCrossAttention(nn.Module): 
    """
    Simple Multi head attention with einops and F.scaled_dot_product_attention.
    """
    def __init__(self, config, is_causal=True):

        super().__init__()

        assert config.n_heads == config.n_kv_heads, "n_heads should be equal n_kv_heads"

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads # for simplicity we use vanilla transformer.
        self.repeats = self.n_heads // self.n_kv_heads

        self.qw = nn.Linear(config.dim, config.head_dim * config.n_heads, bias=False)
        self.kw = nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=False)
        self.vw = nn.Linear(config.dim, config.head_dim * config.n_kv_heads, bias=False)

        self.project = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
        # self.block_size = config.block_size

        self.kv_cache = None

    def forward(self, x, context, attn_mask=None, use_kv_cache=None):
        """
        context should have the same dim as x vectors. 
        context seqlen >> x seqlen
        """
        B, T, C = x.size() # b, t, c*h 
        q, k, v = self.qw(x), self.kw(context), self.vw(context) 

        # split by n_heads.
        q = rearrange(q, 'b t (nh c) -> b nh t c', nh = self.n_heads)
        k = rearrange(k, 'b t (nh c) -> b nh t c', nh = self.n_heads)
        v = rearrange(v, 'b t (nh c) -> b nh t c', nh = self.n_heads)

        if attn_mask is not None:
            t_q, t_k = q.size(2), k.size(2)
            attn_mask = attn_mask[..., -t_q:, -t_k:]
            
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  

        res = rearrange(res, 'b h t c -> b t (h c)')
        res = self.project(res)
        return res    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, rope=None, kv_cache=False):
        x = x + self.attn(self.ln_1(x), attn_mask, rope, kv_cache=kv_cache) 
        x = x + self.mlp(self.ln_2(x))
        return x
    
class CrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_block = Block(config)

        self.ln_1 = nn.LayerNorm(config.dim)
        self.cross_attn = CausalCrossAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x, context, self_attn_mask=None, cross_attn_mask=None, sa_rope=None):
        """
        Block with Cross attention -> Block with Self Attention. 
        """
        # cross attention
        x = x + self.cross_attn(self.ln_1(x), context, attn_mask=cross_attn_mask) 
        x = x + self.mlp(self.ln_2(x))
        
        # self attention
        x = self.sa_block(x, attn_mask=self_attn_mask, rope=sa_rope)

        return x

## Models.
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_size = config.patch_size 
        self.n_electrodes = config.n_electrodes
        self.n_patches_per_channel = config.window_size // config.patch_size

        self.block_size = self.n_patches_per_channel * config.n_electrodes # number of all tokens
        
        self.to_patches = Rearrange('b (t p1) c -> b (t c) p1', p1=self.patch_size)

        self.transformer = nn.ModuleDict(dict(
            emb = nn.Linear(config.patch_size, config.dim),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.dim),
        ))

        self.space_embedding = nn.Parameter(torch.randn(1, config.n_electrodes, config.dim), requires_grad=True)
    
        # self.class_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.precompute_rope_cash = build_complex_rope_cache(dim=self.config.head_dim,
                                                             seq_len=self.block_size,
                                                             theta=config.rope_theta)


        self.register_buffer('attn_mask', build_advanced_causal_mask(block_size=self.block_size,
                                                                    tok_per_time=self.n_electrodes))

        print("Encoder: number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print('Shape of casual mask: ', self.attn_mask.shape)
        print('Shape of the rope cache: ', self.precompute_rope_cash.shape)

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

    @property
    def spatial_pos_embedding(self):
        """
        Returns spatial positional tokens.
        # 1, n_tokens, dim
        """
        space = self.space_embedding.repeat((1, self.n_patches_per_channel, 1))
        return space
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, x, kv_cache=None):
        """
        myo signals with shape: with shape [B, T, C]
        """

        x = self.to_patches(x) # [b, t, n_ch] -> [b, (t/patch * n_ch) , patch]

        b, n_tokens, c = x.shape 
        # embedding
        x = self.transformer.emb(x)
        x = x + self.spatial_pos_embedding[:, -n_tokens:]
        
        for block in self.transformer.h:
            x = block(x, 
                      attn_mask=self.attn_mask, 
                      rope=self.rope_cache, 
                      kv_cache=kv_cache)

        x = self.transformer.ln_f(x)
        return x 

class MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder_dim = config.decoder_dim
        
        self.encoder = Encoder(config)

        self.decoder = nn.ModuleDict(dict(
            emb = nn.Identity(), # nn.Linear(config.dim, config.decoder_dim), # connection between them.
            h = nn.ModuleList([Block(config) for _ in range(config.n_dec_layers)]),
        ))

        self.mask_token = nn.Parameter(torch.randn(config.dim))

        self.decoder_pos_emb = nn.Embedding(self.encoder.block_size, config.decoder_dim)

        self.to_signals = nn.Linear(config.decoder_dim, config.patch_size)
        self.to_signal_shape = Rearrange('b (t c) p -> b (t p) c', c=config.n_electrodes, p=config.patch_size)

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
    
    def get_sub_att_matrix(self, attn_mask, unmasked_indices):
        """
        We have to prepare indexes in the following format.
        att_mask - T, T 
        unmasked_indices - [B, N]
        # B, 1, 1
        # B, N, 1
        # B, 1, N

        Returns: b 1 h w'

        """

        b, n_unmasked = unmasked_indices.shape
        
        ## Getting sub attention mask 
        batch_range = torch.arange(b, device=attn_mask.device)[:, None, None]
        orig_att_mask = attn_mask.expand(b, -1, -1) # 64, 64  -> b, 64, 64
        att_mask_submatrix = orig_att_mask[batch_range, unmasked_indices[..., None], unmasked_indices[:, None, :]] 
        att_mask_submatrix = rearrange(att_mask_submatrix, 'b h w -> b 1 h w')     

        return att_mask_submatrix

    def forward(self, x, targets=None, date_info=None, masking_ratio=0.75, return_preds=False):
        """
        Inputs: x with shape -> B, T, C
        """
        b, t, c = x.shape

        x = self.encoder.to_patches(x)
        b, n_tokens, _ = x.shape

        masked_indices, unmasked_indices = self.get_masking_indices(masking_ratio, x) # [b, N]
        batch_range = torch.arange(b, device=x.device)[:, None]
        
        ## Prepare positional embedding, rope cache and causal attn matrices.
        # 1. expand to batch size.
        spatial_pos_emb = self.encoder.spatial_pos_embedding.expand(b, -1, -1)
        rope_cache = self.encoder.rope_cache.expand(b, -1, -1)

        # 2. get only unmasked indices. they different for each sample in batch
        spatial_pos_emb = spatial_pos_emb[batch_range, unmasked_indices]
        rope_cache = rope_cache[batch_range, unmasked_indices]

        # 3. get sample specific attn causal mask. 
        attn_mask = self.get_sub_att_matrix(self.encoder.attn_mask, unmasked_indices)

        ## ENCODER

        tokens = x[batch_range, unmasked_indices]
        tokens = self.encoder.transformer.emb(tokens)

        tokens = tokens + spatial_pos_emb

        for block in self.encoder.transformer.h:
            tokens = block(tokens, attn_mask=attn_mask, rope=rope_cache)
        
        tokens = self.encoder.transformer.ln_f(tokens)


        ## DECODER 
        unmasked_decoder_tokens = self.decoder.emb(tokens)

        decoder_tokens = torch.zeros(b, n_tokens, self.decoder_dim, device=x.device, dtype=x.dtype)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = self.mask_token

        decoder_pos_emb = self.decoder_pos_emb(torch.cat([unmasked_indices, masked_indices], 1))
        decoder_tokens = decoder_tokens + decoder_pos_emb

        for block in self.decoder.h:
            decoder_tokens = block(decoder_tokens)

        ## LOSS: only on masked tokens.
        ## masked_indices should be refined. Remove padded indices HERE

        mask_tokens = decoder_tokens[batch_range, masked_indices]
        pred_singal_values = self.to_signals(mask_tokens)
        
        # calculate reconstruction loss
        masked_tokens = x[batch_range, masked_indices]
        recon_loss = F.mse_loss(pred_singal_values, masked_tokens)

        if return_preds:

            # what we masked?
            binary_mask = torch.zeros_like(x, device=x.device, dtype=x.dtype) 
            binary_mask[batch_range, masked_indices] = 1

            reconstruction_signal = torch.zeros_like(x, device=x.device, dtype=x.dtype)
            reconstruction_signal[batch_range, masked_indices] = pred_singal_values
            reconstruction_signal[batch_range, unmasked_indices] = x[batch_range, unmasked_indices]

            return recon_loss, self.to_signal_shape(reconstruction_signal), self.to_signal_shape(binary_mask)
        return (recon_loss, None)

class BrainFormer(nn.Module): 
    config = Config
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.encoder = Encoder(config.encoder)
        self.n_output_tokens = config.n_output_tokens

        self.learnable_queries = nn.Parameter(torch.zeros(1, config.n_output_tokens, config.dim))
        self.perceiver = nn.ModuleDict(dict(
                h = nn.ModuleList([CrossBlock(config) for _ in range(config.n_layers)]),
                ln_f = nn.LayerNorm(config.dim), 
                to_motion = nn.Linear(config.dim, config.output_dim))
        )
        
        self.register_buffer('cross_attn_mask', None)
        self.register_buffer('self_attn_mask', None)

        self.precompute_rope_cash = build_complex_rope_cache(dim=config.head_dim,
                                                             seq_len=config.n_output_tokens,
                                                             theta=config.rope_theta)

        print("Full HandFormer: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
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
    
    def forward(self, x, targets=None, date_info=None):
        """
        Get forward pass with loss calculation.
        Inputs: 
        x
            shape b t c 
        targets:
            B, C, T
        """
        b, t, c = x.shape

        emg_context = self.encoder(x) # b, n_tokens, dim
        
        input = self.learnable_queries.expand(b, self.n_output_tokens, -1)
        
        for cross_block in self.perceiver.h:
            input = cross_block(input, emg_context, self.self_attn_mask, 
                                self.cross_attn_mask, sa_rope = self.rope_cache)
        
        pred = self.perceiver.ln_f(input)
        pred = self.perceiver.to_motion(pred)

        if targets is None:
            return None, pred
        
        loss = F.l1_loss(pred, targets)
        return loss, pred
    
    @torch.no_grad()    
    def inference(self, myo, date_info):
        """
        x (signal) - Time, Channel
        OUTPUTS - Time//8, N_BONES
        """
        x = torch.from_numpy(myo)
        t, c = x.shape
        x = rearrange(x, 't c -> 1 t c', t=t, c=c)
        x = x.to(self.device).to(self.dtype)

        pred = self.forward(x, targets=None)
        pred = pred[0].detach().cpu().numpy()

        return pred.T



@torch.no_grad()
def default_generation(model, emg, stride=8):
    """
    emg - [Time, n_channels]
    """
    T = emg.size(0)
    ws = model.config.window_size
    
    n_iters = int((T-ws)//stride)


    sample = emg[:ws]
    for i in range(n_iters):
        inp = sample[None, ...]
        res = model(inp)

        start = int(i * stride)
        sample = emg[start : start+ws]
    
    return 'Completed'

@torch.no_grad()
def cache_generation(model, emg, stride=8):
    """
    emg - [Time, n_channels]
    """
    T = emg.size(0)
    ws = model.config.window_size

    
    n_iters = int((T-ws)//stride)


    sample = emg[:ws]
    for i in range(n_iters):
        res = model(sample[None, ...], use_kv_cache=True)

        start = int(i * stride) + ws - stride
        sample = emg[start:start+stride]
    
    return 'Completed'         



# from latency_test_models import estimate_latency


# if __name__ == '__main__':
#     dtype = torch.float32
#     device = 'cuda'

#     B, C, T = 4, 8, 256
#     INPUTS = torch.zeros(B, C, T).to(device).to(dtype)
#     TARGETS = torch.zeros(B, 20, T//8).to(device).to(dtype)
#     print('INPUTS', INPUTS.shape)
#     print('TARGETS', TARGETS.shape)


#     ## Train MAE
#     mae_config = MAEConfig()
#     mae_model = MAE(mae_config)
#     mae_model = mae_model.to(device).to(dtype).eval()
#     loss = mae_model(INPUTS)
#     print(loss)


#     ## Finetune HandFormer
#     config = HandFormerConfig(encoder=mae_config)
#     model = HandFormer(config)
#     model.encoder = mae_model.encoder 
#     model = model.to(device).to(dtype).eval()

#     MYO = np.zeros([32, C])
#     hat = model.inference(MYO)
#     print(hat.shape)
#     estimate_latency(model, x=MYO, dtype=dtype, device=device)

