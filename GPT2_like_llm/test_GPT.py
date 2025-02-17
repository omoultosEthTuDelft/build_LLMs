import torch
import torch.nn as nn
import tiktoken
from torchinfo import summary

from GPT_model import GPTModel
from GPT2_config import GPT_CONFIG_124M




if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')
    batch = []
    txt1 = 'Every effort moves you'
    txt2 = 'Every day holds a'
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    batch = batch.to(torch.long)

    model = GPTModel(GPT_CONFIG_124M)
    output = model(batch)

    print(f'\nInput batch\n{batch}')
    print(f'\nOutput shape\n[batch_size, num_token, vocab_size]:')
    print(f'{output.shape}')
    print(f'\nOutput\n{output}\n')

    # print(f'model details:{model}')
    # print(model.state_dict)
    
    print('Metrics using "summary" from torchmetrics')
    summary(model, input_size=(batch.shape[0], batch.shape[1])) 

    
    print()
    print('Custom metrics by me:')
    
    # Parameter computation (with and without "weight tying")
    total_params = sum(p.numel() for p in model.parameters())
    total_params_gpt2 = (total_params - sum(p.numel() for p in model.out_head.parameters()))

    # Memory requirements
    total_size_bytes = total_params * 4 # assuming float32 * 4 bytes/parameter
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(90*'=')
    print(f'Total number of parameters: {total_params}')
    print(f'Number of trainable parameters (considering weight tying): {total_params_gpt2}')

    print(f'Token embedding layer shape: {model.tok_emb.weight.shape}')
    print(f'Output layer shape: {model.out_head.weight.shape}')

    print(f'Estimated memory requirements (MB): {total_size_mb:.2f}')
    print(90*'=')


