import torch
import torch.nn as nn

from layer_normalization_class import LayerNorm
from transformer import TransformerBlock
from GPT2_config import GPT_CONFIG_124M


class GPTModel(nn.Module):
    """GPT2-type architecture (O. Moultos - Feb 7):

    Constructor elements 
    1.1. Initialization of token and positional embeddings 
         - self.tok_emb: converts input token indices into dense embeddings
         - self.pos_emb: adds positional embeddings to retain order information
    1.2. First dropout, before entering the transformer block
         - self.drop_emb: applies dropout to embeddings to prevent overfitting
    1.3. Splat operator calling n_layers x transformer block
         - self.trf_blocks: sequence of transformers containing multi-head attention,
           feedforward, normalization, and dropout layers (see TransformerBlock class)
    1.4. Final normalization to standardize the outputs of the transformer
         - self.final_norm: layer normalization for training stabilization
    1.5. Linear output unbiased head to project the output of the transformer
         into the vocabulary space of the tokenizer to generate logits for each
         token of the vocabulary.
         - self.out_head: linear layer mapping hidden states back to vocabulary 
           logits for prediction
    
    Forward pass
    2.1. Extract batch size and sequence length
    2.2 - 2.3 Computes token (each token is mapped to a multi-D tensor) and positional 
         (proviced each token with a unique position representation) embeddings
    2.4  Summation of token and positional embeddings so each token can carry
         both a token identity and position information
    2.5. Applies dropout to the sum of token and positional embeddings
    2.6. Passes embeddings through the transformer layers
    2.7. Applies final normalization
    2.8. Computes logits for the next-token prediction (unnormalized probabilities 
         to be converted to tokens and text output)
    """
    def __init__(self, cfg): # receives a dictionary with the configuration
        super().__init__()
        # 1.1
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        # 1.2
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        # 1.3
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        # 1.4
        self.final_norm = LayerNorm(cfg['emb_dim']) 
        # 1.5
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        in_idx = in_idx.to(torch.long) # I ensure this is of dtype long for use in torchmetrics
        # 2.1
        batch_size, seq_len = in_idx.shape
        # 2.2
        tok_embeds = self.tok_emb(in_idx)
        # 2.3
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # 2.4
        x = tok_embeds + pos_embeds
        # 2.5
        x = self.drop_emb(x)
        # 2.6
        x = self.trf_blocks(x)
        # 2.7
        x = self.final_norm(x)
        # 2.8
        logits = self.out_head(x)
        # logits = self.out_head(self.final_norm(self.trf_blocks(self.drop_emb(tok_embeds + pos_embeds))))
        return logits