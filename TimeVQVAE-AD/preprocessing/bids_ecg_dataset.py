import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import mne
from preprocessing.preprocess import set_window_size

def scale(x, return_scale_params: bool = False):
    """
    instance-wise scaling.
    global-scaling is not used because there are time series with local-mean-shifts such as "UCR_Anomaly_sddb49_20000_67950_68200.txt".

    :param x: (n_convariates, window_length) or it can be (batch, n_convariates, window_length)
    :return: scaled x
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    # centering
    mu = torch.nanmean(x, dim=-1, keepdim=True)
    x = x - mu

    # var scaling
    std = torch.std(x, dim=-1, keepdim=True)  # (n_covariates, 1)
    min_std = 1.e-4  # to prevent numerical instability in scaling.
    std = torch.clamp(std, min_std, None)  # same as np.clip; (n_covariates, 1)
    x = x / std

    if return_scale_params:
        return x, (mu, std)
    else:
        return x
    
class SeizeIT2ECGSequence(Dataset):
    """
    A PyTorch Dataset that:
      - Reads all EEG/ECG EDF files for train/test
      - Extracts the ECG channel
      - Builds sliding windows of length `window_size`
      - Applies per-window scaling (zero-mean, unit-variance)
      - (If test) builds a binary mask window for seizure vs normal
    """

    def __init__(self, bids_root: str, split: str, config: dict, dataset_idx: int):
        self.window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
        base = Path(bids_root) / f"{dataset_idx:03d}_SeizeIT2-ECG"
        ecg_files = sorted(base.glob(f"*_{split}_ecg.edf"))

        # For test, annotation files with same stem but .tsv
        ann_files = [edf.with_suffix('.tsv') for edf in ecg_files] if split=='test' else [None]*len(ecg_files)

        X_windows = []
        Y_windows = [] if split=='test' else None

        for edf_path, ann_path in zip(ecg_files, ann_files):
            # 1) Load and pick ECG channel
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            ecg = raw.copy().pick_channels(['ECG']).get_data().squeeze()  # shape: (n_samples,)

            # 2) If test, build a binary mask over the full series
            if split=='test':
                sf = raw.info['sfreq']
                mask_full = np.zeros_like(ecg, dtype=int)
                ann = np.loadtxt(str(ann_path), delimiter='\t',
                                 dtype=[('on','f4'),('off','f4')])
                for on, off in ann:
                    start, stop = int(on * sf), int(off * sf)
                    mask_full[start:stop] = 1
            else:
                mask_full = None

            # 3) Slide windows manually
            L = len(ecg)
            for i in range(0, L - self.window_size + 1):
                win = ecg[i : i + self.window_size].astype(np.float32)
                x = torch.from_numpy(win)[None, :]        # shape (1, window_size)
                x = scale(x)                              # zero-mean, unit-var
                X_windows.append(x)

                if split=='test':
                    mwin = mask_full[i : i + self.window_size]
                    Y_windows.append(torch.from_numpy(mwin.astype(np.int64)))

        # stack into big tensors
        self.X = torch.stack(X_windows)                 # (N_windows, 1, window_size)
        self.Y = torch.stack(Y_windows) if split=='test' else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is None:
            return self.X[idx]
        return self.X[idx], self.Y[idx]