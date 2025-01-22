import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # x^1
     [0.55, 0.87, 0.66],  # x^2
     [0.57, 0.85, 0.64],  # x^3
     [0.22, 0.58, 0.33],  # x^4
     [0.77, 0.25, 0.10],  # x^5
     [0.05, 0.80, 0.55]   # x^6
     ]
     )

# print(inputs.shape)
# print(inputs.shape[0])
# print(torch.empty(inputs.shape[0]))

# for x^2 as query
query = inputs[1] # to grab the 2nd input token as query
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product of the query with every other input token

print(attn_scores_2)

# simple - x_i / Sum of x - normalization of the attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f'\nAttention weights: {attn_weights_2_tmp}')
print(f'Sum: {attn_weights_2_tmp.sum()}')

# normalization with softmax fn (not great implementation)
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# better version for numerical stability 
# def softmax_naive(x):
#     # Subtract max value for numerical stability
#     x_exp = torch.exp(x - x.max(dim=-1, keepdim=True).values)
#     return x_exp / x_exp.sum(dim=-1, keepdim=True)

attn_weights_naive = softmax_naive(attn_scores_2)
print(f'\nAttention weights: {attn_weights_naive}')
print(f'Sum: {attn_weights_naive.sum()}')

# and by using pytorch's softmax...
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f'\nAttention weights: {attn_weights_2}')
print(f'Sum: {attn_weights_2.sum()}')



