import torch
import torch.nn as nn
import tiktoken
from torchinfo import summary

from GPT_model import GPTModel
from GPT2_config import GPT_CONFIG_124M
from generate_text_simple import generate_text_simple, text_to_token_ids, token_ids_to_text



if __name__ == '__main__':
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = 'Every effort moves you'
    tokenizer = tiktoken.get_encoding('gpt2')

    token_ids = generate_text_simple(model=model, 
                                     idx=text_to_token_ids(start_context, tokenizer),
                                     max_new_tokens=10,
                                     context_size=256) #GPT_CONFIG_124M["context_length"])
    
    print(f'Output text:\n{token_ids_to_text(token_ids, tokenizer)}')

    