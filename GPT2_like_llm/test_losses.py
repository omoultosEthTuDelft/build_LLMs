import torch
import torch.nn as nn

from tokenize_text import tokenize_text
from dataloader_tokenize import create_dataloader_v1
from GPT2_config import GPT_CONFIG_124M
from calculate_loss import calc_loss_loader
from GPT_model import GPTModel



if __name__ == '__main__':
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    TRAIN_RATIO = 0.90
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    text_data, vocab = tokenize_text(DOC_PATH)
    split_idx = int(TRAIN_RATIO * len(text_data))
    train_data = text_data[:split_idx]
    validation_data = text_data[split_idx:]
    # print(validation_data)

    train_loader = create_dataloader_v1(
        txt=train_data,
        batch_size=2,
        max_length=256, #GPT_CONFIG_124M['context_length'],
        stride=256, #GPT_CONFIG_124M['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    validation_loader = create_dataloader_v1(
        txt=validation_data,
        batch_size=2,
        max_length=256, #GPT_CONFIG_124M['context_length'],
        stride=256, #GPT_CONFIG_124M['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print('\nTrain loader')
    for x, y in train_loader: print(x.shape, y.shape)

    print('\nValidation loader')
    for x, y in validation_loader: print(x.shape, y.shape)

    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        validation_loss = calc_loss_loader(validation_loader, model, device)
    
    print(f'Training loss: {train_loss}  |  Validation loss: {validation_loss}')
    






