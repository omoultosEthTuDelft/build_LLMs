import torch
import torch.nn as nn
import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):  # idx(batch, n_tokens)
    for _ in range(max_new_tokens):  # loops for max_new_tokens to be generated
        idx_cond = idx[:, -context_size:]  # crops the context size if it exceeds the supported context size 
        with torch.no_grad():
            logits = model(idx_cond) 

        logits = logits[:, -1, :]   # focuses only on the last time step
        probs = torch.softmax(logits, dim=-1) # probs has shape (batch, vocab_size) - logits to probabilities
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # idx_next has shape (batch, 1) - selects the arg with the highest probability
        idx = torch.cat((idx, idx_next), dim=1)    # idx has shape (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adds batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes batch dimension
    return tokenizer.decode(flat.tolist())
