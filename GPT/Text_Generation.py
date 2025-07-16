import torch 
import torch.nn as nn
from GPT.Tokenization import text_to_tokens, token_to_text

# ============ Greedy Decoding ==========================
def generate_simple_text(model, idx, max_new_tokens, context_size):
    # idx (batch, n_tokens) array of indices in the current index 
    for _ in range(max_new_tokens):
        # Cropping the current context if it exceeds the supported context size 
        # eg: if LLM supports only 5 tokens, and the context size is 10 then only the last 5 tokens are used as context 
        idx_cond = idx[:, -context_size:]

        # Get the predictions 
        with torch.no_grad():
            logits = model(idx_cond) # => batch, n_tokens, vocab_size 
        
        # Focus only on last time step 
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply the softmax to get probabilites 
        probas = torch.softmax(logits, dim=-1) # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value 
        idx_next = torch.argmax(probas, dim = -1, keepdim = True) # (batch, 1)

        # Append sampled index to the running sequence 
        idx = torch.cat((idx, idx_next), dim = 1) # (batch, n_tokens + 1)

    return idx

def generate_and_print_sample(model, tokenizer, start_context, device):
    model.eval()
    context_size = model.pos_emb.weight.shape[0] 
    encoded = text_to_tokens(start_context, tokenizer = tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_simple_text(model=model, idx = encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()

# ================ Top_K + Temperature Scaling ======================== 
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k = None, eos_id = None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling 
        if top_k is not None:
            # Keep only top_K values 
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        
        # Apply Temperature Scaling 
        if temperature > 0.0:
            logits = logits / temperature 
            # Apply the softmax to get probabilities 
            probs = torch.softmax(logits, dim = -1) # (batch_Size, context_len)
            # Sample from distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

        # Otherwise same as before : get idx of the vocab entry with the highest logits value 
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim=True) # (batch_size, 1)
        
        if idx_next == eos_id: # Stop Generating early if end_of_sequence is encountered and eos is specified
            break

        # Same as before: Append sample index to the running sequence 
        idx = torch.cat((idx, idx_next), dim = 1) # (batch_size, num_tokens + 1)

    return idx
