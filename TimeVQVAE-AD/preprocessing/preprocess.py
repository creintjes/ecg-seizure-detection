import re
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

from collections import OrderedDict
import psutil

# Match filenames like: sub-001_run-03_preprocessed.pkl
pattern = re.compile(r"sub-(\d+)_run-(\d+)_preprocessed\.pkl$")
def seconds_to_samples(sec, fs): return int(round(sec * fs))

def scale(x, return_scale_params: bool = False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    mu = torch.nanmean(x, dim=-1, keepdim=True)
    x = x - mu
    std = torch.std(x, dim=-1, keepdim=True)   # changed: nan-aware
    std = torch.clamp(std, 1e-4, None)
    x = x / std
    return (x, (mu, std)) if return_scale_params else x


@dataclass
class ASIMAnomalySequence:
    subject_id: str
    run_id: str
    data: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_path(
        cls, 
        path: Path, 
        expand_labels: bool = False, 
        sampling_rate: int = None,
        pre_minutes: float = 5.0,
        post_minutes: float = 3.0
    ) -> 'ASIMAnomalySequence':
        match = pattern.match(path.name)
        assert match, f"Filename {path.name} doesn't match expected pattern."
        subject_id, run_id = match.groups()

        with open(path, "rb") as f:
            raw = pickle.load(f)

        data = raw['channels'][0]['windows'][0]
        labels = np.array(raw['channels'][0]['labels'][0])

        assert len(data) == len(labels), "Data and labels must have the same length."

        if expand_labels:
            assert sampling_rate is not None, "sampling_rate must be provided if expand_labels is True"
            labels = cls._expand_labels(
                labels, 
                sampling_rate, 
                pre_minutes=pre_minutes, 
                post_minutes=post_minutes
            )

        return cls(
            subject_id=subject_id,
            run_id=run_id,
            data=np.array(data),
            labels=labels
        )

    @staticmethod
    def _expand_labels(labels, sampling_rate, pre_minutes=5.0, post_minutes=3.0):
        expanded = labels.copy()
        n = len(labels)
        pre_samples = int(pre_minutes * 60 * sampling_rate)
        post_samples = int(post_minutes * 60 * sampling_rate)
        ones = np.where(labels == 1)[0]
        for idx in ones:
            start = max(0, idx - pre_samples)
            end = min(n, idx + post_samples + 1)
            expanded[start:end] = 1
        return expanded



class ASIMAnomalyDataset(Dataset):
    def __init__(self,
                 kind: str,
                 data_dir: Path,
                 window_size: int,
                 stride: int = 1,
                 sub: str = "all",
                 sampling_rate: int = None,
                 expand_labels: bool = False,
                 max_memory_gb: float = 12.0,
                 pre_minutes: float = 5.0, 
                 post_minutes: float = 3.0,
                 batch_size: int = 256):
        if stride == -1:
            if kind == 'train':
                stride = seconds_to_samples(24, sampling_rate)
            else:
                stride = seconds_to_samples(2, sampling_rate)
        assert stride > 0, "stride must be >= 1"
        if expand_labels:
            assert sampling_rate is not None, "sampling_rate is required when expand_labels=True"
        
        self.kind = kind
        self.window_size = window_size
        self.stride = stride
        self.expand_labels = expand_labels
        self.sampling_rate = sampling_rate
        self.pre_minutes = pre_minutes
        self.post_minutes = post_minutes
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        # Store file paths instead of loaded sequences
        if sub == "all":
            sub_re = re.compile(r"^sub-(0(?:0[1-9]|[1-8][0-9]|9[0-6]))_run-(\d+)_preprocessed\.pkl$")

            self.file_paths = sorted(
                p for p in data_dir.glob("sub-*_run-*_preprocessed.pkl")
                if sub_re.match(p.name)  # only subs 001…096
            )
        else:
            self.file_paths = sorted(data_dir.glob(f"sub-{sub}_run-*_preprocessed.pkl"))

        if not self.file_paths:
            raise ValueError(f"No matching files found for sub={sub} in {data_dir}")
        
        sample_path = self.file_paths[0]
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
            seq_size = len(sample_data['channels'][0]['windows'][0]) * 4  # 4 bytes per float32

        suggested_cache = batch_size * 4  # Double batch size for buffer
        memory_based_cache = int(self.max_memory_bytes * 0.8 / seq_size)
        self.cache_size = min(suggested_cache, memory_based_cache)
        print(f"Caching up to {self.cache_size} sequences (using ~{self.cache_size * seq_size / 1024**3:.1f}GB RAM)")
        # Initialize LRU cache
        self.cache = OrderedDict()
        
        # Pre-compute metadata without loading full sequences
        self.indices = self._build_indices()
        
    def _load_sequence(self, file_path):
        """Lazy load sequence and update cache"""
        if file_path in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(file_path)
            return self.cache[file_path]
            
        # Load new sequence
        sequence = ASIMAnomalySequence.from_path(
            file_path,
            expand_labels=self.expand_labels,
            sampling_rate=self.sampling_rate,
            pre_minutes=self.pre_minutes,
            post_minutes=self.post_minutes
        )
        
        # Update cache
        self.cache[file_path] = sequence
        if len(self.cache) > self.cache_size:
            # Remove least recently used
            self.cache.popitem(last=False)
            
        return sequence

    def _build_indices(self):
        indices = []
        for seq_idx, file_path in enumerate(self.file_paths):
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
                data = raw['channels'][0]['windows'][0]
                labels = np.array(raw['channels'][0]['labels'][0])
                T = len(data)
            if T < self.window_size:
                continue  # skip too-short sequences

            if self.kind == 'train' and self.expand_labels:
                labels = ASIMAnomalySequence._expand_labels(
                    labels, self.sampling_rate,
                    pre_minutes=self.pre_minutes, post_minutes=self.post_minutes  # NEW
                )

            added = 0
            for start in range(0, T - self.window_size + 1, self.stride):
                if self.kind == 'train':
                    if labels[start:start + self.window_size].sum() == 0:
                        indices.append((seq_idx, start))
                        added += 1
                else:
                    indices.append((seq_idx, start))
                    added += 1
            if self.kind == 'train' and added == 0:
                # Optional: warn if a run contributes no normal windows after dilation
                print(f"[warn] no normal windows after dilation for {file_path.name}")
        return indices

    def __getitem__(self, idx):
        seq_idx, start = self.indices[idx]
        file_path = self.file_paths[seq_idx]
        
        # Lazy load sequence
        sequence = self._load_sequence(file_path)
        
        # Get window
        x = sequence.data[start:start + self.window_size, None]
        x = rearrange(x, 'l c -> c l')
        x = torch.from_numpy(x).float()
        x = scale(x)

        if self.kind == 'train':
            return x
        else:
            y = sequence.labels[start:start + self.window_size]
            y = torch.from_numpy(y).long()
            return x, y

    def __len__(self):
        return len(self.indices)

    def memory_usage(self):
        """Monitor memory usage"""
        process = psutil.Process()
        return f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB"