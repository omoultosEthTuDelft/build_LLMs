import torch
from torch.utils.data import Dataset, DataLoader
from dataloader_tokenize import GPTDatasetV1, create_dataloader_v1


# print(torch.__version__)
# print(tiktoken.__version__)

if __name__ == '__main__':

    # General idea/pipeline: Input text -> Tokenized text -> Token IDs -> Token embeddings + Positional embeddings ->Input embeddings


    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    vocab_size = 50257 # which is the vocab size of the BPE tokenizer we used
    output_dim = 256   # number of dimensions for the encoding of input tokens
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    MAX_LENGTH = 4  # number of tokens in each batch
    BATCH = 8 # number of text samples

    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=BATCH, 
        max_length=MAX_LENGTH, 
        stride=MAX_LENGTH,  # sliding window size for parsing the text
        shuffle=False   
    ) 

    data_iter = iter(dataloader) # to convert the dataloader into a Python iterator
    inputs, targets = next(data_iter)
    print(f'Token IDs:\n{inputs}')
    print(f'\n* Inputs shape: {inputs.shape}')

    token_embeddings = token_embedding_layer(inputs) # each token ID is now embedded as 256-D tensor
    print(f'* Shape of token embedding layer: {token_embeddings.shape}') # resulting shape is 8 x 4 x 256 (batch x max_length x encoding dim)

    # creating another embedding layer with the same embedding dimension as the one in line 35
    context_length = MAX_LENGTH
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f'* Shape of position embedding layer: {pos_embeddings.shape}')

    input_embeddings = token_embeddings + pos_embeddings # the embedded input examples that can now be processed by the main LLM modules
    print(f'* Shape of token + position embeddings: {input_embeddings.shape}')