"""
Creates a character-level language-modelling dataset from a stream of text.
You shouldn't need to make any changes to this file.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, config, data, type='test', train_dataset = None):
        self.config = config
        self.data = data
        self.vocab_size = None

        if type == 'test':
            self.train_dataset_ref = train_dataset        
        elif type == 'train':
            self.train_dataset_ref = self
            chars = sorted(list(set(data)))
            data_size, vocab_size = len(data), len(chars)
            print('data has %d characters, %d unique.' % (data_size, vocab_size))

            self.stoi = { ch:i for i,ch in enumerate(chars) }
            self.itos = { i:ch for i,ch in enumerate(chars) }
            self.vocab_size = vocab_size
        

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.train_dataset_ref.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y