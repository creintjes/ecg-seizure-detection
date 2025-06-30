import os
import pickle
from typing import List, Tuple
import numpy as np
from pathlib import Path

def load_preprocessed_samples(data_dir: str, max_loaded_files: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load preprocessed ECG window samples from pickle files and collect them into a list.

    Args:
        data_dir: Path to directory containing the preprocessed .pkl files.

    Returns:
        List of windowed ECG samples as numpy arrays and a List with their according labels [0 or 1].
    """
    all_samples: List[np.ndarray] = []
    all_labels: List[int] = []
    amount_empty_or_corrupted_files: int = 0
    count_loaded_files : int = 0

    # Iterate through all .pkl files in the given directory
    for filename in os.listdir(data_dir):
        
        if count_loaded_files >= max_loaded_files:
            break

        if filename.endswith("_preprocessed.pkl"):
            filepath = os.path.join(data_dir, filename)
            
            # Skip empty files
            if os.path.getsize(filepath) == 0:
                print(f"Skipped empty file: {filename}")
                amount_empty_or_corrupted_files += 1
                continue

            try:
                with open(filepath, 'rb') as file:
                    data = pickle.load(file)

                    if not data or "channels" not in data:
                        continue

                    # Iterate through each ECG channel
                    for channel_data in data["channels"]:
                        windows = channel_data.get("windows", [])
                        labels = channel_data.get("labels", [])
                        all_samples.extend(windows)
                        all_labels.extend(labels)
                    count_loaded_files +=1
            
            except (EOFError, pickle.UnpicklingError) as e:
                # print(f"Warning: {filename} is empty or corrupted.")
                print(f"Corrupted pickle file: {filename} ({e})")
                amount_empty_or_corrupted_files += 1
                continue
    print(f"Amount empty or corrupted files {amount_empty_or_corrupted_files}.")
    return all_samples, all_labels

import pickle
import numpy as np
from pathlib import Path

def load_continuous_test_series(data_dir: str, dataset_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the full, un-windowed ECG series and binary label mask for `dataset_idx`.
    Expects you to have saved one file named like:
      {dataset_idx:03d}_continuous_test.pkl
    containing a dict with keys 'ecg' (1D np.array) and 'mask' (1D np.array of 0/1).
    """
    p = Path(data_dir) / f"{dataset_idx:03d}_continuous_test.pkl"
    print("path", p)
    with open(p, 'rb') as f:
        d = pickle.load(f)
    # d['ecg']: shape (ts_len,), d['mask']: shape (ts_len,)
    return d['ecg'].astype(np.float32), d['mask'].astype(np.int64)