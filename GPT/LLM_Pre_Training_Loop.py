import torch 
import torch.nn as nn 
import tiktoken 
import tqdm 
from torch.utils.data import Dataset, DataLoader 
from GPT.Tokenization import token_to_text, text_to_tokens
from GPT.Loss_Calculation import calc_loss_batch, calc_loss_loader
from GPT.Text_Generation import generate_and_print_sample, generate_simple_text

# ================ Training Loop =============================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # List to track losses and seen tokens 
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Training Loop 
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train() # Training mode 

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients 
            optimizer.step() # Update model weights using loss gradients 
            tokens_seen += input_batch.numel() # Returns the total number of element (or tokens) in the input_batch
            global_step += 1

            # Evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Epoch: {epoch + 1} (Step: {global_step:06d}) | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        
        # Print sample text after each epoch 
        generate_and_print_sample(
            model, tokenizer, start_context, device
        )

    return train_losses, val_losses, track_token_seen