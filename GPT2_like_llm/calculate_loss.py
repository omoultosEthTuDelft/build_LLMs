import torch
import torch.nn as nn


def calc_loss_batch(input_batch, target_batch, model, device):
    """Utility function that calculates the cross entropy loss of
       a given batch returned via a training and validation loader."""
    
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()  # flattened tensors by combining over the 
        )                                             #  batch dimension i.e., batch x number of tokens
    
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Function which uses the calc_loss_batch utility function to
       compute the loss over all the batches sampled by a given data loader.
       This function iterates over all batches in a given data loader, accumulates 
       the loss in the total_loss variable, and then computes and averages the 
       loss over the total number of batches."""
    
    total_loss = 0.
    if len(data_loader) == 0: return float('nan')
    elif num_batches is None: num_batches = len(data_loader)
    else: num_batches = min(num_batches, len(data_loader)) # Reduces the number of batches to match the 
                                                           # total number of batches in the data
                                                           # loader if num_batches exceeds the number
                                                           # of batches in the data loader

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()   # Sums loss for each batch
        else:
            break
    
    return total_loss / num_batches    # Average over all batches

