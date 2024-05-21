## Models.
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

from .blocks import Block, build_complex_rope_cache, RMSNorm, build_advanced_causal_mask
from simple_parsing import Serializable
from dataclasses import dataclass


class CausalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.dim),
        ))


        # for default causal forward generation
        self.precompute_rope_cash = build_complex_rope_cache(dim=self.config.head_dim,
                                                             seq_len=config.block_size,
                                                             theta=config.rope_theta)
        
        self.attn_mask = torch.tril(torch.ones(config.block_size, config.block_size)).to(torch.bool)


        print('Shape of the rope cache: ', self.precompute_rope_cash.shape)

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        myo signals with shape: with shape [B, T, C]
        """
        attn_mask = self.attn_mask if attn_mask is None else attn_mask
        rope_cache = self.rope_cache if rope_cache is None else rope_cache

        attn_mask = attn_mask.to(self.device)

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



class Franky(nn.Module): 
    """This is first model which incorporate brain features into LLM"""

    def __init__(self, combiner_config, causal_config, brain_model, llm_model, tokenizer=None):
        super().__init__()
        self.brain_model = brain_model
        self.llm_model = llm_model
        self.tokenizer = tokenizer


        self.combiner_config = combiner_config
        self.causal_config = causal_config

        self.n_electrodes = brain_model.config.n_electrodes
        self.window_size = self.brain_model.config.window_size
        self.block_size = int(self.window_size / self.brain_model.tokenizer.downsample * self.combiner_config.n_registers)
        self.dim = combiner_config.dim
        
        # self.combiner_ln = RMSNorm(combiner_config.dim)
        # self.combiner_pos_embeddings = nn.Parameter(torch.randn(1, self.n_electrodes, combiner_config.dim) / 0.02)
        # self.combiner_model = nn.Sequential(*[Block(combiner_config) for _ in range(combiner_config.n_layers)])
        
        
        # Causal model
        # init new rope cache for working with several registers and overwriting old one
        # we have to repeat values, because n_registers have same time step
        causal_config.block_size = self.block_size
        
        # self.causal_model = CausalModel(causal_config)
        # self.causal_model.attn_mask = build_advanced_causal_mask(self.block_size, self.combiner_config.n_registers)
        # old_rope = self.causal_model.precompute_rope_cash
        # self.causal_model.precompute_rope_cash = old_rope.repeat_interleave(self.combiner_config.n_registers, dim=0)

        self.projector = nn.Linear(brain_model.config.dim, llm_model.config.n_embd, bias=True)
        
        self.date_embeddings = nn.Embedding(num_embeddings=25, embedding_dim=llm_model.config.n_embd)


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
    
    # Применяем только к комбайнеру

    # def _init_weights(self, module):

    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def combine_features(self, x):
        """
        Combine features from different channels into several vectors.
        x: b, t, c, d
        """
        b, t, c, d = x.size()
        x = rearrange(x, 'b t c d -> (b t) c d', b=b, t=t, c=self.n_electrodes, d=self.dim)

        x = x + self.combiner_pos_embeddings
        x = self.combiner_model(x)

        x = x[:, :self.combiner_config.n_registers]
        x = self.combiner_ln(x)
        x = rearrange(x, '(b t) r d -> b (t r) d', b=b, t=t, r=self.combiner_config.n_registers, d=self.brain_model.dim)
        return x 
    
    
    def forward(self, x, targets=None, date_info=None):
        """
        Train model.
        x: B, T, C
        """
        is_padded = (x==0).all(dim=-1) # B, T

        # tmp = int(self.brain_model.tokenizer.downsample / self.combiner_config.n_registers)
        tmp = int(self.brain_model.tokenizer.downsample)
        is_padded = is_padded[:, ::tmp]

        _, x = self.brain_model(x) # b, t, c, d

        latents = torch.mean(x, dim=2)
        # x = self.combine_features(x)
        # latents = self.causal_model(x)

        # Also we have to add padded tokens here. and do not calculate metrics on them.
        # future_loss = F.mse_loss(pred_latent[:, :-self.combiner_config.n_registers], x[:, :-self.combiner_config.n_registers])
        
        features = self.projector(latents)  

        # date_embedding = self.date_embeddings(date_info)
        # x = torch.cat([x, date_embedding], dim=-1)
        
        new_idx = targets.clone()
        new_idx[new_idx == -100] = self.tokenizer.eos_token_id
        

        # print('INPUT', new_idx[:, :-1])
        # print('TARGETS', targets[:, 1:])
        # print('~is_padded', ~is_padded, is_padded.shape)
        
        outputs = self.llm_model(input_ids=new_idx[:, :-1], 
                                 labels=targets[:, 1:], 
                                 encoder_hidden_states=features, 
                                 encoder_attention_mask=~is_padded)
        
        return outputs.loss, outputs.logits

    
    @torch.no_grad()
    def generate(self, x, date_info=None, tokenizer=None):
        device = self.device
        
        x = torch.from_numpy(x[None, ]).to(device).to(self.dtype)
        features = self.brain_model(x)
        
        ### Text part
        start = tokenizer.bos_token
        input_ids = tokenizer(start,  return_tensors="pt")['input_ids'].to(self.device)

        # max_new_tokens = 25
        # temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        # top_k = 10

        res = self.llm_model.generate(input_ids=input_ids, encoder_hidden_states=features)
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