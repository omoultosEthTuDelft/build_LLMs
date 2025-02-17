import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

# x = np.arange(-100, 100)
x = torch.linspace(-3, 3, 100)
# print(x)

relu_np = np.maximum(0, x)
relu = nn.ReLU()
y_relu = relu(x)
# print(y_relu == relu_np)

gelu_np = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
# gelu_np = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 1 * x**3)))

gelu = nn.GELU()
y_gelu = gelu(x)
# print(y_gelu == gelu_np)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([gelu_np, y_gelu, y_relu], ["GELU_custom", "torch GELU", "torch ReLU"]), 1):
    plt.subplot(1, 3, i)
    plt.plot(x, y)
    plt.title(f'{label} activation fn')
    plt.xlabel('x')
    plt.ylabel(f'{label}(x)')
    plt.grid(True)
plt.tight_layout()
plt.show()

# plt.plot(x, relu, label='ReLU')
# plt.plot(x, gelu, label='GELU')
# plt.xlabel('x')
# plt.ylabel('Activation')
# plt.legend()
# plt.show()