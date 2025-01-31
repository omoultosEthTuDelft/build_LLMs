import torch

# dummy input
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

def attn_weights_context_vectors_input_2_only():
    # for x^2 as query
    query = inputs[1] # to grab the 2nd input token as query
    attn_scores_2 = torch.empty(inputs.shape[0])
    context_vec_2 = torch.zeros(query.shape) # to create the context vector z^2 (this corresponds only to )

    # tokens -> inputs -> attention weights for each token -> context vector for each token
    print()
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) # dot product of the query with every other input token
        print(f'Attention scores for 2nd token [{i}] =  {x_i} x {query} = {attn_scores_2[i]}')

    print()
    print(80*'*')
    print(f'  Attention scores a: {attn_scores_2}')
    print(80*'*')


    # simple - x_i / Sum of x - normalization of the attention scores
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    # print(f'\nAttention weights: {attn_weights_2_tmp}')
    # print(f'Sum: {attn_weights_2_tmp.sum()}')

    # normalization with softmax fn (not great implementation)
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)

    # better version for numerical stability 
    # def softmax_naive(x):
    #     # Subtract max value for numerical stability
    #     x_exp = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    #     return x_exp / x_exp.sum(dim=-1, keepdim=True)

    attn_weights_naive = softmax_naive(attn_scores_2)
    # print(f'\nAttention weights: {attn_weights_naive}')
    # print(f'Sum: {attn_weights_naive.sum()}')

    # and by using pytorch's softmax...
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    # print(f'\nAttention weights using softmax: {attn_weights_2}')
    # print(f'Sum: {attn_weights_2.sum()}')
    print()
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
        print(f'Context vector for 2nd token {attn_weights_2[i]} x {x_i} = {context_vec_2}')

    print()
    print(58*'*')
    print(f'  Context vector z^(2): {context_vec_2}')
    print(58*'*')


def attention_weights_for_loop(inputs = inputs):
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    attn_weights = torch.softmax(attn_scores, dim=1)
    return attn_scores, attn_weights

def attention_weights_mat_mul(inputs = inputs):
    attn_scores = inputs @ inputs.T  #= np.matmul(inputs, inputs.T)
    attn_weights = torch.softmax(attn_scores, dim=-1)  # dim = -1 will softmax the last dimension
    check_normalization = attn_weights.sum(dim=-1) # check if all rows sum to 1 (i.e., validate normalization)
    context_vectors = attn_weights @ inputs
    return attn_scores, attn_weights, check_normalization, context_vectors
    
def print_attn():
    scores, weights, check, context_vec = attention_weights_mat_mul()
    print(f'\n\nInformation on the shapes of the matrices\n{80*"-"}')
    print(f'Shape of inputs: {inputs.shape} | Shape of attention scores: {scores.shape}')
    print(f'Shape of attention weights: {weights.shape} | Shape of context vectors: {context_vec.shape}')
    print(f'\nattentions scores (inputs @ inputs.T)\n{60*"-"}\n{scores}')
    print(f'\nattention weights (softmax normalized attn scores)\n{60*"-"}\n{weights}')
    print(f'\nSum of each column of attention weights: {check}')
    print(f'\nContext vectors (attn weights @ inputs)\n{40*"-"}\n{context_vec}')


if __name__ == '__main__':
    # attn_weights_context_vectors_input_2_only()
    # print(attention_weights_for_loop())
    # print(attention_weights_for_loop == attention_weights_mat_mul)
    print_attn()