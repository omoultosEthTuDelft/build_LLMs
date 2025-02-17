import torch
import torch.nn as nn
# import numpy as np

# Simple Neural Network with 5 inputs and 6 outputs
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example) 
print()
print(f'Input\n{batch_example}')
print()
print(f'Output\n{out}')

mean = out.mean(dim=-1, keepdim=True)  # dim=-1 computes the mean across the column dimension -> one mean per row
variance = out.var(dim=-1, keepdim=True)

print(f'\nMean\n{mean} | \n\nVariance\n{variance}')


out_norm = (out - mean) / torch.sqrt(variance)
mean = out_norm.mean(dim=-1, keepdim=True)
variance = out_norm.var(dim=-1, keepdim=True)

print(f'\nNormalized layer outputs:\n{out_norm}')
torch.set_printoptions(sci_mode=False)
print(f'\nAbsolute mean:\n{torch.abs(mean)}')
print(f'\nMean:\n{mean}')
print(f'\nVariance:\n{variance}')
