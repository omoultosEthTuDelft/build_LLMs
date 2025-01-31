import torch
import torch.nn as nn

class CasualAttention(nn.Module):
    def __init__(self, 
                 d_in, 
                 d_out, 
                 context_length,
                 dropout,
                 gkv_bias=False
                 ):
        super().__init__()
        self.d_out   = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=gkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=gkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=gkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length), 
            diagonal=1)
            )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        
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
    
    batch = torch.stack((inputs, inputs), dim=0)
    # print(batch.shape)

    DIM_IN = inputs.shape[1] #input embedding size (dim=3)
    DIM_OUT = 2  # output embedding size (dim=2)
    torch.manual_seed(123)
    CONTEXT_LEN = batch.shape[1]

    ca = CasualAttention(DIM_IN, DIM_OUT, CONTEXT_LEN, 0.0)
    context_vecs = ca(batch)
    print(f'context_vecs.shape: {context_vecs.shape}')
    print(f'context_vecs: \n{context_vecs}')
