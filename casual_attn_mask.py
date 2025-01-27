import torch
from self_attn_class_v2 import SelfAttention_v2 as sa_v2

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
torch.manual_seed(789)
sa_v2 = sa_v2(DIM_IN, DIM_OUT)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
