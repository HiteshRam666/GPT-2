import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import tiktoken

# ========== Dataset & DataLoaders =====================
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
    
        # Tokenize the entire text 
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # Sliding window to chunk the book into overlapping sequences of max_length 
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, 
                         stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    # Initialize the tokenizer 
    tokenizer = tiktoken.get_encoding('gpt2')

    # Create Dataset 
    dataset = GPTDataset(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)

    # Create DataLoader 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader
