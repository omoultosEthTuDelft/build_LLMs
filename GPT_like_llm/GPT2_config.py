# def gpt_config():
GPT_CONFIG_124M = {
    'vocab_size'    : 50257,    # Vocabulary size
    'context_length': 1024,     # Context length, i.e., max numbrer of input tokens the model can handle via positional embeddings
    'emb_dim'       : 768,      # Embedding dimension, i.e., embedding size, transforming each token into a 768-dimensional vector
    'n_heads'       : 12,       # Number of attention heads in the multi-head attention mechanism
    'n_layers'      : 12,       # Number of layers, i.e., number of transformer blocks in the model
    'drop_rate'     : 0.1,      # Dropout rate, 0.1 = 10% tensor data dropout to prevent overfitting
    'qkv_bias'      : False     # Query-Key-Value bias, i.e., bias vector in the Linear layers of multi-head attention for QKV computations
}

    # return GPT_CONFIG_124M
