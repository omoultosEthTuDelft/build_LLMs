import torch
import torch.nn as nn   

from feed_forward_class import FeedForward
from layer_normalization_class import LayerNorm
from multi_head_attn_wrapper_weight_splits import MultiHeadAttention
from GPT2_config import GPT_CONFIG_124M



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadAttention(                     # Multi-head attn 
            d_in =           cfg['emb_dim'],
            d_out =          cfg['emb_dim'],
            context_length = cfg['context_length'],
            dropout=         cfg['drop_rate'],
            num_heads =      cfg['n_heads'],
            qkv_bias =       cfg['qkv_bias'])
        self.ff = FeedForward(cfg)                          # Feed-forward
        self.norm1 = LayerNorm(cfg['emb_dim'])              # Layer-normalization
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])   # Dropout


    def forward(self, x):
        # Shortcut connection for attn block
        shortcut = x
        x = self.drop_shortcut(self.attn(self.norm1(x)))
        x = x + shortcut  # Addition of the original input 

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.drop_shortcut(self.ff(self.norm2(x)))
        x = x + shortcut  # Addition of the original input

        return x
    


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)   # sample input of shape: [batch_size=2, num_tokens=4, emb_dim=768]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print(f'\nInput shape: {x.shape}')
    print(f'Output shape: {output.shape}')



    