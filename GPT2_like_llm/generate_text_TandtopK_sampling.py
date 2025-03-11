import torch
import torch.nn as nn
import tiktoken

def generate_text(model, idx, max_new_tokens, context_size,
                  temperature=0.0, top_k=None, eos_id=None):  # eos = end of sequence token id
    for _ in range(max_new_tokens):  # loops for max_new_tokens to be generated
        idx_cond = idx[:, -context_size:]  # crops the context size if it exceeds the supported context size 
        with torch.no_grad():
            logits = model(idx_cond) 

        logits = logits[:, -1, :]   # focuses only on the last time step

        # filters logits with top_k sampling 
        if top_k is not None:  
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # temperature scaling 
        if temperature > 0.0:  
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1 , keepdim=True)
        
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adds batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes batch dimension
    return tokenizer.decode(flat.tolist())
