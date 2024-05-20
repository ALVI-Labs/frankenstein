



class Franky(nn.Module): 
    """This is first model which incorporate brain features into LLM"""

    def __init__(self, brain_model, llm_model, tokenizer=None, combiner_config):
        super().__init__()

        self.brain_model = brain_model
        self.combiner = nn.Sequential(*[Block(combiner_config) for _ in range(combiner_config.n_layers)])
        
        self.projector = nn.Linear(self.brain_model.config.dim, llm_model.config.n_embd)
        self.llm_model= llm_model
        self.tokenizer = tokenizer
        
        

        self.date_embeddings = nn.Embedding(num_embeddings=25, embedding_dim=llm_model.config.n_embd)
        
        print("Full Franky: number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def combine_features(self, x):
        b, t, c, d = x.size()
        tokens = rearrange(tokens, 'b t c d -> (b t) c d', b=b, t=t, c=self.brain_model.n_electrodes, d=self.brain_model.dim)
        tokens = self.combiner(x)
        tokens = tokens[:, -2:]
        tokens = rearrange(tokens, '(b t) c d -> b (c t) d', b=b, t=t, c=2, d=self.brain_model.dim)
        return tokens 
    def causal_modelling(self, x):
        return x 
    
    def forward(self, x, targets=None, date_info=None):
        """
        Train model.
        """

        _, x = self.brain_model(x)
        x = self.combine_features(x)
        x = self.causal_modelling(x)
        
        features = self.projector(x)
    
        new_idx = targets.clone()
        new_idx[new_idx == -100] = self.tokenizer.eof_token

        outputs = self.llm_model(input_ids=new_idx[:, :-1], 
                                 labels=targets[:, 1:], 
                                 encoder_hidden_states=features, 
                                 encoder_attention_mask=None)
        
        return outputs.loss, outputs.logits

    
    @torch.no_grad()
    def generate(self, x, date_info=None, tokenizer=None):
        device = self.device
        
        x = torch.from_numpy(x[None, ]).to(device).to(self.dtype)
        features = self.brain_model(x)
        
        ### Text part
        start = '<|endoftext|>'
        input_ids = tokenizer(start,  return_tensors="pt")['input_ids'].to(self.device)

        # max_new_tokens = 25
        # temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        # top_k = 10

        res = gpt2_cross.generate(input_ids=input_ids, encoder_hidden_states=features)
        pred = gpt2_tokenizer.batch_decode(res)
        
        return pred