import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys     = x @ self.W_key
        querries = x @ self.W_query
        values   = x @ self.W_value

        attn_scores  = querries @ keys.T   # omegas
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        context_vec  = attn_weights @ values

        return context_vec
            
            

if __name__ == '__main__':

    inputs = torch.FloatTensor(
    [[0.43, 0.15, 0.89],  # x^1
     [0.55, 0.87, 0.66],  # x^2
     [0.57, 0.85, 0.64],  # x^3
     [0.22, 0.58, 0.33],  # x^4
     [0.77, 0.25, 0.10],  # x^5
     [0.05, 0.80, 0.55]   # x^6
     ])
    
    DIM_IN = inputs.shape[1] #input embedding size (dim=3)
    DIM_OUT = 2  # output embedding size (dim=2)

    
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in = DIM_IN, d_out = DIM_OUT)
    print(sa_v1(inputs))
