import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np

# import your existing loader
from data_helpers import load_preprocessed_samples  

class PreprocessedSamplesDataset(Dataset):
    """
    Loads ALL windows + labels via load_preprocessed_samples(), 
    then optionally splits into train/test and returns:
      - train:  x only
      - test:   (x, y)
    where x has shape (1, window_length) and y is 0/1.
    """
    def __init__(self,
                 data_dir: str,
                 max_loaded_files: int,
                 kind: str = 'train',
                 train_frac: float = 0.8):
        """
        data_dir:            path to folder with *_preprocessed.pkl files  
        max_loaded_files:    how many .pkl files to read (for debugging)  
        kind:                'train' | 'test' | 'all'  
        train_frac:          fraction of samples to put in train split  
        """
        assert kind in ('train','test','all')
        # load raw numpy windows + labels lists
        samples, labels = load_preprocessed_samples(data_dir, max_loaded_files)
        
        # stack into tensors
        # samples: List[np.ndarray(window_length,)]
        # →  X: (n_windows, window_length)
        X = torch.stack([torch.from_numpy(w) for w in samples]).float()
        # add channel dim → (n_windows, 1, window_length)
        self.X = X.unsqueeze(1)

        # labels: List[int]
        self.Y = torch.tensor(labels, dtype=torch.long)

        # train/test split
        n = len(self.X)
        split = int(n * train_frac)
        if kind == 'train':
            self.X = self.X[:split]
        elif kind == 'test':
            self.X = self.X[split:]
            self.Y = self.Y[split:]
        # if kind=='all', we keep everything

        self.kind = kind

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]               # (1, window_length)
        # instance‐wise normalization:
        mu  = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-4)
        x = (x - mu) / std

        if self.kind in ('train','all'):
            return x
        else:  # 'test'
            return x, self.Y[idx]