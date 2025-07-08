import re
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange


# Match filenames like: sub-001_run-03_preprocessed.pkl
pattern = re.compile(r"sub-(\d+)_run-(\d+)_preprocessed\.pkl$")


def scale(x, return_scale_params: bool = False):
    """
    Instance-wise standardization.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    mu = torch.nanmean(x, dim=-1, keepdim=True)
    x = x - mu

    std = torch.std(x, dim=-1, keepdim=True)
    min_std = 1e-4
    std = torch.clamp(std, min_std, None)
    x = x / std

    if return_scale_params:
        return x, (mu, std)
    else:
        return x


@dataclass
class ASIMAnomalySequence:
    subject_id: str
    run_id: str
    data: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_path(cls, path: Path) -> 'ASIMAnomalySequence':
        match = pattern.match(path.name)
        assert match, f"Filename {path.name} doesn't match expected pattern."
        subject_id, run_id = match.groups()

        with open(path, "rb") as f:
            raw = pickle.load(f)

        data = raw['channels'][0]['windows'][0]
        labels = raw['channels'][0]['labels'][0]

        assert len(data) == len(labels), "Data and labels must have the same length."

        return cls(
            subject_id=subject_id,
            run_id=run_id,
            data=np.array(data),
            labels=np.array(labels)
        )

 
class ASIMAnomalyDataset(Dataset):
    def __init__(self,
                 kind: str,
                 data_dir: Path,
                 window_size: int,
                 stride: int = 1,
                 sub: str = "all"):
        assert kind in ['train', 'test'], f"Invalid kind: {kind}"
        self.kind = kind
        self.window_size = window_size
        self.stride = stride

        # Load relevant files
        self.sequences = []

        if sub == "all":
            files = sorted(data_dir.glob("sub-*_run-*_preprocessed.pkl"))
        else:
            files = sorted(data_dir.glob(f"sub-{sub}_run-*_preprocessed.pkl"))

        for file in files:
            self.sequences.append(ASIMAnomalySequence.from_path(file))

        if not self.sequences:
            raise ValueError(f"No matching files found for sub={sub} in {data_dir}")

        self.X, self.Y = self._gather_all_data()
        self.indices = self._select_indices()

    def _gather_all_data(self):
        all_x = []
        all_y = []
        for seq in self.sequences:
            x = seq.data[:, None]   # (T, 1)
            y = seq.labels          # (T,)
            all_x.append(x)
            all_y.append(y)
        return all_x, all_y

    def _select_indices(self):
        valid = []
        for seq_idx, (x, y) in enumerate(zip(self.X, self.Y)):
            T = len(x)
            starts = list(range(0, T - self.window_size + 1, self.stride))

            # Add final window if not already included
            last_start = T - self.window_size
            if starts and starts[-1] != last_start:
                starts.append(last_start)

            for i in starts:
                window_labels = y[i:i + self.window_size]
                if self.kind == 'train':
                    if np.all(window_labels == 0):
                        valid.append((seq_idx, i))
                elif self.kind == 'test':
                    valid.append((seq_idx, i))

        return valid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx, start = self.indices[idx]
        x = self.X[seq_idx][start:start + self.window_size]
        y = self.Y[seq_idx][start:start + self.window_size]

        x = rearrange(x, 'l c -> c l')
        x = torch.from_numpy(x).float()
        x = scale(x)

        if self.kind == 'train':
            return x
        else:
            y = torch.from_numpy(y).long()
            return x, y