import mne
import numpy as np

def load_ecg_edf(path: str, picks: str = 'ecg'):
    """
    Load an EDF file and return the ECG channel data and sampling frequency.
    """
    raw = mne.io.read_raw_edf(path, preload=True, stim_channel=None, verbose=False)
    # pick only ECG channel(s)
    raw.pick_channels([ch for ch in raw.ch_names if picks in ch.lower()])
    data, sfreq = raw.get_data(return_times=False), raw.info['sfreq']
    # data shape: (n_channels, n_samples)
    return data.astype(np.float32), float(sfreq)

