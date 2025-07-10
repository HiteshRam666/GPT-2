import tiktoken 
import torch
import torch.nn as nn

# =========== Tokenization ===================
tokenizer = tiktoken.get_encoding('gpt2')

def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension 
    return encoded_tensor 

def token_to_text(token, tokenizer):
    flat = token.squeeze(0) # Remove batch dimension 
    return tokenizer.decode(flat.tolist())