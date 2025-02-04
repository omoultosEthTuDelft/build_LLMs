import torch 
import torch.nn as nn

class DummyDNN(nn.Module):
    """Dummy DNN to showcase the use of shortcut connections, i.e., core building block of
    LLMs because they are important with the vanishing gradient problem in DNNs."""
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)  # output of the current layer
            print(f'layer: {layer}') 
            print(f'layer_output: {layer_output}, shape of layer_output: {layer_output.shape}, shape of x: {x.shape}')
            if self.use_shortcut and x.shape == layer_output.shape:  # checks if shortcut should be applied
                print(f'-----> shape of layer_output: {layer_output.shape}, shape of x: {x.shape}')
                print(f'x={x}')
                x = x + layer_output
            else:
                x = layer_output
        print(150*"-")
        return x
    

def print_gradients(model, x):
    output = model(x)  # Forward pass
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target) # Calculates the loss

    loss.backward() # Backward pass to compute the gradients

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')

    
if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1] 
    sample_input = torch.tensor([[1., 0. , -1]])
    torch.manual_seed(123)
    # model_without_shortcut = DummyDNN(layer_sizes=layer_sizes, use_shortcut=False)
    # print('test without shortcuts')
    # print_gradients(model_without_shortcut, sample_input)
    # print()

    print(f'\n{20*"-"}')
    print(f' test with shortcuts \n{150*"-"}') 
    model_with_shortcut = DummyDNN(layer_sizes=layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

