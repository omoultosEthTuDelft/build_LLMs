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

context_length = attn_scores.shape[0]
# torch.tril makes a tensor with values above the diagonal are zero
mask_simple = torch.tril(torch.ones(context_length, context_length)) 
print(mask_simple)

masked_simple = attn_weights * mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
print(row_sums)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)


# Dropout in deep learning is a technique where randomly selected hidden layer units
# are ignored during training, effectively “dropping” them out. This method helps prevent
# overfitting by ensuring that a model does not become overly reliant on any specific
# set of hidden layer units. When applying dropout to an attention weight matrix with a rate of 50%, half of the
# elements in the matrix are randomly set to zero. To compensate for the reduction in
# active elements, the values of the remaining elements in the matrix are scaled up by a
# factor of 1/0.5 = 2. This scaling is crucial to maintain the overall balance of the attention weights, 
# ensuring that the average influence of the attention mechanism remains
#consistent during both the training and inference phases.

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout of 50%
example = torch.ones(6, 6)
print(example)
print(dropout(example))
torch.manual_seed(123)
print(dropout(attn_weights))