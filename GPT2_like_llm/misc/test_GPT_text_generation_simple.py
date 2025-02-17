import torch
import torch.nn as nn
import tiktoken
from torchinfo import summary

from GPT_model import GPTModel
from GPT2_config import GPT_CONFIG_124M
from generate_text_simple import generate_text_simple



if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')
    
    state_context = 'Hello, I am'
    encoded = tokenizer.encode(state_context)
    print(f'\nencoded: {encoded}')

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adds batch dimension
    print(f'encoded_tensor.shape: {encoded_tensor.shape}')

    model = GPTModel(GPT_CONFIG_124M)

    model.eval()
    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M['context_length'])

    print(f'\nOutput length: {len(out[0])}')    
    print(f'\n\nOutput: \n{out}')
    

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f'\nDecoded output: \n{decoded_text}')

