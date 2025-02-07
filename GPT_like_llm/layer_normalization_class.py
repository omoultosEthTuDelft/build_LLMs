import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """The main idea behind layer normalization (not the same as batch normalization) is to 
    adjust the activations (outputs) of a neural network layer to have a mean of 0 and a 
    variance of 1 (unit variance). This adjustment speeds up the convergence to effective 
    weights and ensures consistent, reliable training. In GPT-2 and modern transformer
    architectures, layer normalization is typically applied before and after the multi-head
    attention module."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable parameters of the same dimension as the input
        self.shift = nn.Parameter(torch.zeros(emb_dim))
       
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased controls if we apply Bessel's correction or not
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # to prevent division with 0
        
        return self.scale * norm_x + self.shift
    

if __name__ == '__main__':
    
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    torch.set_printoptions(sci_mode=False)
    print(f'Mean:\n{mean}')
    print(f'Variance:\{var}')