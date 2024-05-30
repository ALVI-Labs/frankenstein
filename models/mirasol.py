## Models.
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

from .blocks import Block, build_complex_rope_cache, RMSNorm, build_advanced_causal_mask
from simple_parsing import Serializable
from dataclasses import dataclass


@dataclass
class MirasolConfig(Serializable):
    # data params
    window_size: int = 512
    n_electrodes: int = 256
    mask_ratio: float = 0.75

    n_registers: int = 6

    n_layers: int = 12
    dim: int = 512
    hidden_dim: int = 2048

    head_dim: int = 32
    n_heads: int = 16
    n_kv_heads: int = 16 # now it should be the same with n_heads.

    dropout: float = 0.1
    rope_theta: float = 10000.0

    w_latent_loss: float = 1.0
    w_recon_loss: float = 1.0

class CausalModel(nn.Module):
    """
    Apply causal attention with RoPE and Attention mask. Same rope for N tokens per time step.
    B, N, D -> B, N, D
    """
    def __init__(self, config, block_size, num_tokens_per_time=1):
        super().__init__()
        self.config = config
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            # ln_f = RMSNorm(config.dim),
        ))

        # for default causal forward generation
        rope = build_complex_rope_cache(dim=self.config.head_dim,
                                        seq_len=block_size / num_tokens_per_time,
                                        theta=config.rope_theta)

        self.precompute_rope_cash = rope.repeat_interleave(num_tokens_per_time, dim=0)
        self.attn_mask = build_advanced_causal_mask(block_size=block_size, tok_per_time=num_tokens_per_time)

        print('Shape of the rope cache: ', self.precompute_rope_cash.shape)
        print('Shape of the causal model: ', self.attn_mask.shape)



    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        myo signals with shape: with shape [B, T, C]
        """
        attn_mask = self.attn_mask if attn_mask is None else attn_mask
        attn_mask = attn_mask.to(x.device)
        
        rope_cache = self.rope_cache if rope_cache is None else rope_cache

        # embedding
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask, rope=rope_cache)
        return x
    
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

class Combiner(nn.Module):
    """
    Combine inputs features into fixed number of embeddings 
    Add learnabe positional embeddings, blocks, get n_registers as combined representation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_registers = config.n_registers

        self.pe = nn.Parameter(torch.randn(1, config.n_electrodes, config.dim))
        
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            # ln_f = RMSNorm(config.dim),
        ))

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        Transform embeddings: B, N, D -> B, M, D
        """

        x = x + self.pe
        x = self.transformer.drop(x)
        
        for block in self.transformer.h:
            x = block(x)
        
        return x[:, :self.n_registers]
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
class Reconstructor(nn.Module):
    """
    Get combined n features and reconstruct all features. 
    Add learnabe positional embeddings. No dropout.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_registers = config.n_registers

        self.cls_tokens = nn.Parameter(torch.randn(1, config.n_electrodes, config.dim))
        self.pe = nn.Parameter(torch.randn(1, config.n_electrodes + config.n_registers, config.dim))
        
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.dim),
        ))

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        Transform: B, M, D -> B, N, D
        """
        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe
        
        # embedding
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return x[:, :self.config.n_electrodes]
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

class Mirasol(nn.Module): 
    """This is first model which incorporate brain features into LLM"""

    def __init__(self, config, vq_model):
        super().__init__()
        self.vq_model = vq_model
        self.config = config
        self.tokenizer_downsample = vq_model.downsample
        self.window_size= config.window_size
        self.block_size = self.get_block_size()

        
        # use embeddings from vq vae and then linear transform them.
        self.proj_emb = nn.Linear(vq_model.D, config.dim)
        # self.emb = nn.Embedding(vq_model.codebook_size, config.dim)
        
        self.combiner = Combiner(config)
        self.drop_combiner_out = nn.Dropout1d(p=config.mask_ratio)

        self.causal = CausalModel(config, block_size=self.block_size,
                                  num_tokens_per_time=config.n_registers)
        self.proj_to_next_token = nn.Linear(config.dim, config.dim)
        
        self.reconstructor = Reconstructor(config)
        self.proj_to_indices = nn.Linear(config.dim, vq_model.codebook_size)
        
        self.w_latent_loss = config.w_latent_loss
        self.w_recon_loss = config.w_recon_loss
        self.apply(self._init_all_weights)
        
        print("Full Mirasol model: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_block_size(self):
        block_size = self.config.window_size / self.vq_model.downsample * self.config.n_registers
        return int(block_size)
    
    def cosine_loss(self, x, y, is_padded):
        valid_mask = ~is_padded

        cosine_sim = F.cosine_similarity(x, y, dim=-1, eps=1e-05)
        cosine_sim = (cosine_sim * valid_mask).sum() / valid_mask.sum()
        
        cosine_loss = 1 - cosine_sim
        return cosine_loss
    
    def adjust_pad_mask(self, is_padded, scale_factor):
        if scale_factor > 1:
            is_padded = is_padded[:, ::scale_factor]
        else:
            scale_factor = int(1/scale_factor)
            is_padded = torch.repeat_interleave(is_padded, scale_factor, dim=-1)
        return is_padded

    def forward(self, x, targets=None, date_info=None, return_paddings=False):
        """
        Forward pass of the Mirasol model processes input `x` through its components 
        to generate embeddings, apply transformations, and compute losses 
        using cosine similarity and cross-entropy.

        Parameters:
            x (torch.Tensor): Input tensor of shape [B, T, C] where B is the batch size,
                            T is the number of time steps, and C is the number of channels.
            targets (torch.Tensor, optional): Not used in this implementation.
            date_info (Any, optional): Not used in this implementation.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The sum of latent and reconstruction losses.
                - latent (torch.Tensor): The latent embeddings after processing through the causal model.


        t = T/8
        VQ VAE: B, T, C -> B, t, C
        Emb: B, t, C -> B, t, C, D
        Combiner: (B t), C, D -> (B t), M, D
        Causal: B, (t M) , D -> B, (t M) - 1 , D
        Reconstructor: B, (t M)-1 , D -> B, (t M)-1,C

        """
        scale_factor = int(self.window_size // self.block_size)
        
        is_padded = (x==0).all(dim=-1) # B, T
        is_padded = self.adjust_pad_mask(is_padded, scale_factor=scale_factor)

        indices_out, embeds = self.vq_model.get_indices(x, return_embeddings=True) # maybe scale there.
        
        x = self.proj_emb(embeds)
        
        b, t, c, d = x.size()
        x = rearrange(x, 'b t c d -> (b t) c d', b=b, t=t, 
                           c=self.config.n_electrodes, d=self.config.dim)

        x = self.combiner(x)

        x = rearrange(x, '(b t) r d -> b (t r) d', b=b, t=t, 
                           r=self.config.n_registers, d=self.config.dim)
    
        latents = self.causal(self.drop_combiner_out(x)) # b (t r) d -> b (t r) d

        latent_loss = 0 
        if self.w_latent_loss > 0:
            future_hat = self.proj_to_next_token(latents[:, :-self.config.n_registers])
            future = x[:, self.config.n_registers:]
            is_padded_cut = is_padded[:, self.config.n_registers:]
            
            latent_loss = self.cosine_loss(future_hat, future.detach(), is_padded_cut)

        # here we're working with latents
        recon_loss = 0 
        if self.w_recon_loss > 0:
            latents_to_recon = rearrange(latents, ' b (t r) d -> (b t) r d', 
                                         b=b, t=t, r=self.config.n_registers, d=self.config.dim)

            x = self.reconstructor(latents_to_recon) # (b t) r d -> (b t) c d 
            
            preds = self.proj_to_indices(x) # (b t) c d -> (b t) c codebook_size

            preds = rearrange(preds, '(b t) c e -> b t c e', b=b, t=t, 
                              c=self.config.n_electrodes, e=self.vq_model.codebook_size)

            is_padded_idxs = is_padded[:, ::self.config.n_registers]
            indices_out[is_padded_idxs] = -100

            recon_loss = F.cross_entropy(preds[:, :-1].reshape(-1, preds.size(-1)), 
                                        indices_out[:, 1:].reshape(-1).to(torch.long))

        latents = latents[:, :-self.config.n_registers]
        is_padded = is_padded[:, :-self.config.n_registers]

        losses_dict = {'total_loss': self.w_latent_loss * latent_loss + self.w_recon_loss * recon_loss, 
                      'latent_loss': latent_loss,
                      'recon_loss': recon_loss}


        if return_paddings:
            return losses_dict, latents, is_padded

        return losses_dict, latents


    def _init_all_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
class Franky(nn.Module): 
    """
    This is first model which incorporate brain features into LLM
    """

    def __init__(self, brain_model, llm_model, w_txt_loss=1.0, 
                 add_temporal_embeddings=False, num_temporal_embeddings=2048):
        super().__init__()
        
        self.brain_model = brain_model

        self.llm_decoder = llm_model['decoder']
        self.proj_out = llm_model['proj_out']
        self.tokenizer = llm_model['tokenizer']

        n_embd_decoder = self.llm_decoder.config.d_model
        self.projector = nn.Sequential(RMSNorm(brain_model.config.dim), 
                                       nn.Linear(brain_model.config.dim, n_embd_decoder, bias=True))

        self.date_embeddings = nn.Embedding(num_embeddings=25, embedding_dim=n_embd_decoder)

        self.add_temporal_embeddings = add_temporal_embeddings
        if self.add_temporal_embeddings:
            self.wpe = nn.Embedding(num_embeddings=num_temporal_embeddings, embedding_dim=n_embd_decoder)

        self.w_txt_loss = w_txt_loss
        self._init_weights(self.projector)
        self._init_weights(self.date_embeddings)

        print("Full Franky: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, model, mean=0.0, std=0.02):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=mean, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=mean, std=std)
    
    
    def forward(self, x, targets=None, date_info=None):
        """
        Train model.
        x: B, T, C
        """
        losses, latents = self.brain_model(x) # b, N, d

        # fast track for loss calculation.
        if self.w_txt_loss == 0:
            return losses, latents


        features = self.projector(latents)

        if self.add_temporal_embeddings:
            pos = torch.arange(0, features.size(1), dtype=torch.long, device=features.device) # shape (t)
            pos_embeddings = self.wpe(pos)
            features = features + pos_embeddings

        date_embedding = self.date_embeddings(date_info)
        features = torch.cat([features, date_embedding], dim=1)
        
        input_ids = targets.clone()
        input_ids[input_ids == -100] = self.tokenizer.eos_token_id
        
        logits= self.llm_decoder(input_ids=input_ids, 
                                 encoder_hidden_states=features)['last_hidden_state']
        logits = self.proj_out(logits)
        
        txt_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   targets[:, 1:].reshape(-1), ignore_index=-100)
        
        total_loss = losses['total_loss'] + self.w_txt_loss * txt_loss
        losses['total_loss'] = total_loss
        losses['txt_loss'] = txt_loss

        return losses, logits

    
    @torch.no_grad()
    def generate(self, x, date_info=None, tokenizer=None):
        device = self.device
        
        x = torch.from_numpy(x[None, ]).to(device).to(self.dtype)
        _ , features, is_padded = self.brain_model(x, return_paddings=True)
        
        ### Text part
        start = self.tokenizer.bos_token
        input_ids = self.tokenizer(start,  return_tensors="pt")['input_ids'].to(self.device)

        # max_new_tokens = 25
        # temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        # top_k = 10

        res = self.llm_model.generate(input_ids=input_ids, 
                                      encoder_hidden_states=features,
                                      encoder_attention_mask=~is_padded)
        pred = self.tokenizer.batch_decode(res)
        
        return pred

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device