import torch
from torch import nn
import tiktoken

from tokenize_text import tokenize_text
from dataloader_tokenize import create_dataloader_v1
from GPT2_config import GPT_CONFIG_124M
from calculate_loss import calc_loss_batch, calc_loss_loader
from GPT_model import GPTModel
from generate_text_simple import generate_text_simple, text_to_token_ids, token_ids_to_text
from plot_losses import plot_losses 

def train_model_simple(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       num_epochs,
                       eval_freq,
                       eval_iter,
                       start_context,
                       tokenizer):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train() # Put model in the training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'Epoch: {epoch+1} (Step {global_step:06d}): '
                      f'Train loss: {train_loss:.3f} |'
                      f'Val loss: {val_loss:.3f}')

        # Prints a sample text after each epoch        
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model,
                   train_loader,
                   val_loader,
                   device,
                   eval_iter):
    """Calculates the training and validation set losses after each model update so we can evaluate whether
       the training improves the model. More specifically, the evaluate_model function calculates
       the loss over the training and validation set while ensuring the model is in evaluation mode with 
       gradient tracking and dropout disabled when calculating the loss over the training and validation sets"""
    
    model.eval()
    with torch.no_grad():   # Disables gradient tracking
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Function that takes a text snippet (start_context) as input, converts it into token IDs, 
    and feeds it to the LLM to generate a text sample using the generate_text_simple function"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, 
            idx = encoded, 
            max_new_tokens=50, 
            context_size=context_size
            )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()


if __name__=='__main__':
    DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    TRAIN_RATIO = 0.90    
    text_data, vocab = tokenize_text(DOC_PATH)
    split_idx = int(TRAIN_RATIO * len(text_data))
    train_data = text_data[:split_idx]
    validation_data = text_data[split_idx:]
    MODEL = GPTModel(GPT_CONFIG_124M)
    MODEL.to(DEVICE)
    optimizer = torch.optim.AdamW(
        MODEL.parameters(),
        lr=0.0004,
        weight_decay=0.1
        )
    EPOCHS = 10
    TOKENIZER = tiktoken.get_encoding('gpt2')


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

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=MODEL,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=EPOCHS,
        eval_freq=5,
        eval_iter=5,
        start_context='Every effort moves you',
        tokenizer=TOKENIZER
        )

    epochs_tensor = torch.linspace(0, EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)




