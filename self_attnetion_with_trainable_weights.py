import torch

# dummy input
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # x^1
     [0.55, 0.87, 0.66],  # x^2
     [0.57, 0.85, 0.64],  # x^3
     [0.22, 0.58, 0.33],  # x^4
     [0.77, 0.25, 0.10],  # x^5
     [0.05, 0.80, 0.55]   # x^6
     ])

def second_input_element_only():
    x_2 = inputs[1]
    dim_in = inputs.shape[1] #input embedding size (dim=3)
    dim_out = 2  # output embedding size (dim=2)
    
    # initialization of 3 weight matrices, i.e., Wq, Wk, and Wv:
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)  # set to false just for now. For model training -> True
    W_key = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)

    # only for the 2nd input layer
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value

    # for all inputs
    # because the output dim was set to 2, now we project the 6x3 inputs into 6x2 dimensional space
    keys = inputs @ W_key
    values = inputs @ W_value

    # print(x_2, dim_in)
    # print(W_query, W_key, W_value)
    # print(f'shape of x_2={x_2.shape}, shape of W_q={W_query.shape}')
    # print(f'\n\nquery_2 = {x_2} @ {W_query} = {query_2}')

    print(f'\nTshape of inputs: {inputs.shape}')
    print(f'shape of keys tensor: {keys.shape}')
    print(f'shape of values tensor: {values.shape}')



if __name__ == '__main__':
    second_input_element_only()