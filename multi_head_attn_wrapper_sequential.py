import torch
import torch.nn as nn
from casual_attn_class import CasualAttention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, 
                 d_in, 
                 d_out, 
                 context_length,
                 dropout,
                 num_heads,
                 gkv_bias=False
                 ):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, 
                             d_out, 
                             context_length, 
                             dropout, 
                             gkv_bias)
                for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)  # The self attention modules are processed sequentially here
    

if __name__ == '__main__':

    inputs = torch.FloatTensor(
    [[0.43, 0.15, 0.89],  # x^1
     [0.55, 0.87, 0.66],  # x^2
     [0.57, 0.85, 0.64],  # x^3
     [0.22, 0.58, 0.33],  # x^4
     [0.77, 0.25, 0.10],  # x^5
     [0.05, 0.80, 0.55]   # x^6
     ])
    
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    DIM_IN = 3
    DIM_OUT = 2 
    CONTEXT_LEN = batch.shape[1]
    DROPOUT = 0.0
    NUM_HEADS = 2

    mha = MultiHeadAttentionWrapper(DIM_IN, DIM_OUT, CONTEXT_LEN, DROPOUT, NUM_HEADS)
    context_vecs = mha(batch)

    print(f'\nShape of final (concatenated context vector Z: {context_vecs.shape})')
    print(f'\nConcatenated context vector Z\n{60*"-"}\n{context_vecs}\n{60*"-"}')
