import torch
import torch.nn as nn
from GPT2_config import GPT_CONFIG_124M

class FeedForward(nn.Module):
    """The FeedForward module plays a crucial role in enhancing the model’s ability 
    to learn from and generalize the data. Although the input and output dimensions of 
    thismodule are the same, it internally expands the embedding dimension into a 
    higherdimensional space through the first linear layer"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            nn.GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)
    
if __name__ == '__main__':
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)