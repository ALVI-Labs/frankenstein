import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

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
        rope = rope[:T]
    else:
        rope = rope[:, :T]

    rope = rope.unsqueeze(-2) # [B, T, 1, dim//2] or [T, 1, dim//2]

    # b, t, n_h, dim - > (b, t, n_h, dim/2, 2)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))    

    # (b, t, n_h, dim/2, 2) * t, 1, dim/2
    x_out = torch.view_as_real(x_ * rope).flatten(3)
    return x_out.type_as(x)


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

    def forward(self, x, attn_mask=None, rope=None):
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
            attn_mask = attn_mask[..., :t_q, :t_k]

        q = q.transpose(1, 2)  # (B, nh, T, c)
        k = k.transpose(1, 2)  # (B, nh, T, c)
        v = v.transpose(1, 2)  # (B, nh, T, c)
        
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  

        res = rearrange(res, 'b nh t c -> b t (nh c)', b=B, t=T, nh=self.n_heads, c=self.head_dim)
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
        self.ln_1 = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, rope=None):
        x = x + self.attn(self.ln_1(x), attn_mask, rope) 
        x = x + self.mlp(self.ln_2(x))
        return x