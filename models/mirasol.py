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
    window_size: int = 128
    n_electrodes: int = 256
    mask_ratio: float = 0.75

    n_registers: int = 4

    n_layers: int = 4
    dim: int = 64
    hidden_dim: int = 512

    head_dim: int = 16
    n_heads: int = 16
    n_kv_heads: int = 16 # now it should be the same with n_heads.

    dropout: float = 0.1
    rope_theta: float = 10000.0

    w_latent_loss: float = 1.0
    w_recon_loss: float = 1.0

class CausalModel(nn.Module):
    """
    Apply causal attention with RoPE and Attention mask. Same rope for N tokens per time step.
    """
    def __init__(self, config, num_tokens_per_time=1, block_size=256):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.dim),
        ))

        # for default causal forward generation
        rope = build_complex_rope_cache(dim=self.config.head_dim,
                                        seq_len=block_size / 2,
                                        theta=config.rope_theta)
        
        self.precompute_rope_cash = rope.repeat_interleave(num_tokens_per_time, dim=0)
        
        self.attn_mask = build_advanced_causal_mask(block_size=block_size, tok_per_time=num_tokens_per_time)
        print('Shape of the rope cache: ', self.precompute_rope_cash.shape)


    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        myo signals with shape: with shape [B, T, C]
        """
        attn_mask = self.attn_mask if attn_mask is None else attn_mask
        rope_cache = self.rope_cache if rope_cache is None else rope_cache

        # embedding
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask, rope=rope_cache)

        x = self.transformer.ln_f(x)
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
    Apply causal attention with RoPE and Attention mask. Same rope for N tokens per time step.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_registers = config.n_registers

        self.pe =  nn.Parameter(torch.randn(1, config.n_electrodes, config.dim) / 0.02)
        
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.dim),
        ))

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        myo signals with shape: with shape [B, T, C]
        """

        x = x + self.pe
        x = self.transformer.drop(x)
        # embedding
        
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask, rope=rope_cache)

        x = self.transformer.ln_f(x)
        x = x[:, :self.n_registers]

        return x
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
class Reconstructor(nn.Module):
    """
    Apply causal attention with RoPE and Attention mask. Same rope for N tokens per time step.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_registers = config.n_registers

        self.cls_tokens = nn.Parameter(torch.randn(1, config.n_electrodes, config.dim) / 0.02)
        self.pe = nn.Parameter(torch.randn(1, config.n_electrodes + config.n_registers, config.dim) / 0.02)
        
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.dim),
        ))

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        x: B, M, D
        """
        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe
        
        # embedding
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask, rope=rope_cache)

        x = self.transformer.ln_f(x)
        x = x[:, :self.config.n_electrodes]
        return x
    
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



        self.emb = nn.Embedding(vq_model.codebook_size, config.dim)
        self.combiner = Combiner(config)
        self.causal = CausalModel(config, num_tokens_per_time=config.n_registers, block_size=self.block_size)
        self.reconstructor = Reconstructor(config)
        self.linear_to_indices = nn.Linear(config.dim, vq_model.codebook_size)
        
        self.w_latent_loss = config.w_latent_loss
        self.w_recon_loss = config.w_recon_loss
        self.apply(self._init_all_weights)
        
        print("Full Mirasol model: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_block_size(self):
        block_size = self.config.window_size / self.vq_model.downsample * self.config.n_registers
        block_size = int(block_size)
        return block_size
    def cosine_loss(self, x, y, is_padded):
        valid_mask = ~is_padded

        cosine_sim = F.cosine_similarity(x, y, dim=-1)
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
        Forward pass of the Mirasol model processes input `x` through its components to generate embeddings, apply transformations, and compute losses using cosine similarity and cross-entropy.

        Parameters:
            x (torch.Tensor): Input tensor of shape [B, T, C] where B is the batch size,
                            T is the number of time steps, and C is the number of channels.
            targets (torch.Tensor, optional): Not used in this implementation.
            date_info (Any, optional): Not used in this implementation.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The sum of latent and reconstruction losses.
                - latent (torch.Tensor): The latent embeddings after processing through the causal model.


        Next steps
        1. add masking before causal model
        2. calculate loss only on non padded tokens
        3. non padded on reconstructor

        
        VQ VAE: B, T, C -> B, T/8, C
        Emb: B, t, C -> B, t, C, D
        Combiner: (B t), C, D -> (B t), M, D
        Causal: B, (t M) , D -> B, (t M) - 1 , D
        Reconstructor: B, (t M)-1 , D -> B, (t M)-1,C

        """
        scale_factor = int(self.window_size // self.block_size)
        
        is_padded = (x==0).all(dim=-1) # B, T
        is_padded = self.adjust_pad_mask(is_padded, scale_factor=scale_factor)

        indices_out = self.vq_model.get_indices(x)
        tokens = self.emb(indices_out)

        b, t, c, d = tokens.size()

        tokens = rearrange(tokens, 'b t c d -> (b t) c d', b=b, t=t, c=self.config.n_electrodes, d=self.config.dim)

        tokens = self.combiner(tokens)

        tokens = rearrange(tokens, '(b t) r d -> b (t r) d', b=b, t=t, r=self.config.n_registers, d=self.config.dim)
    
        latents = self.causal(tokens) # b (t r) d

        latent_loss = self.cosine_loss(latents[:, :-self.config.n_registers], 
                                       tokens[:, self.config.n_registers:], 
                                       is_padded[:, self.config.n_registers:])

        # here we're working with latents
        recon_loss = 0 
        if self.w_recon_loss > 0:
            latents_to_recon = rearrange(latents, ' b (t r) d -> (b t) r d', b=b, t=t, r=self.config.n_registers, d=self.config.dim)

            x = self.reconstructor(latents_to_recon) # (b t) r d -> (b t) c d 
            
            preds = self.linear_to_indices(x) # (b t) c d -> (b t) c codebook_size

            preds = rearrange(preds, '(b t) c e -> b t c e', b=b, t=t, c=self.config.n_electrodes, e=self.vq_model.codebook_size)

            is_padded_idxs = is_padded[:, ::self.config.n_registers]
            indices_out[is_padded_idxs] = -100

            recon_loss = F.cross_entropy(preds[:, :-1].reshape(-1, preds.size(-1)), 
                                        indices_out[:, 1:].reshape(-1).to(torch.long))

        latents = latents[:, self.config.n_registers:]

        losses_dict = {'total_loss': self.w_latent_loss * latent_loss + self.w_recon_loss * recon_loss, 
                      'latent_loss': latent_loss,
                      'recon_loss': recon_loss}


        if return_paddings:
            return losses_dict, latents, is_padded[:, self.config.n_registers:]

        return losses_dict, latents


    # def _init_weights(self, model, mean=0.0, std=0.02):
    #     for module in model.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.normal_(module.weight, mean=mean, std=std)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.Embedding):
    #             nn.init.normal_(module.weight, mean=mean, std=std)

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
    """This is first model which incorporate brain features into LLM"""

    def __init__(self, brain_model, llm_model, w_txt_loss=1.0):
        super().__init__()
        
        self.brain_model = brain_model

        self.llm_decoder = llm_model['decoder']
        self.proj_out = llm_model['proj_out']
        self.tokenizer = llm_model['tokenizer']

        n_embd_decoder = self.llm_decoder.config.d_model
        self.projector = nn.Linear(brain_model.config.dim, n_embd_decoder, bias=True)

        self.date_embeddings = nn.Embedding(num_embeddings=25, embedding_dim=n_embd_decoder)

        self.w_txt_loss = w_txt_loss

        # self._init_weights(self.combiner_model)
        # self._init_weights(self.causal_model)
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
        features = self.projector(latents)  

        # date_embedding = self.date_embeddings(date_info)
        # x = torch.cat([x, date_embedding], dim=-1)
        
        new_idx = targets.clone()
        new_idx[new_idx == -100] = self.tokenizer.eos_token_id
        
        logits= self.llm_decoder(input_ids=new_idx,
                                 encoder_hidden_states=features)['last_hidden_state']
        logits = self.proj_out(logits)
        txt_loss = F.cross_entropy(logits[:, :-1].view(-1, logits.size(-1)),
                                   new_idx[:, 1:].view(-1))
        

        losses['total_loss'] = losses['total_loss'] + self.w_txt_loss * txt_loss
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

        res = self.llm_model.generate(input_ids=input_ids, encoder_hidden_states=features, encoder_attention_mask=~is_padded)
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