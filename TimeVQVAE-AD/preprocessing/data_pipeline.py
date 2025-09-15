from torch.utils.data import DataLoader
from preprocessing.preprocess import ASIMAnomalyDataset
from pathlib import Path
from utils import set_window_size
    
def build_data_pipeline(batch_size: int,
                       data_dir: Path,
                       sub: str,
                       kind: str,
                       window_size: int,
                       stride: int,
                       num_workers: int,
                       sampling_rate: int,  # Add this parameter
                       n_periods: int = 120,  # Default value for n_periods
                       bpm: int = 75,  # Default value for heartbeats per minute
                       expand_labels: bool = False) -> DataLoader:
    """Build data pipeline."""
    window_size = set_window_size(sampling_rate, n_periods=n_periods, bpm=bpm)

    dataset = ASIMAnomalyDataset(
        kind=kind,
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        sub=sub,
        sampling_rate=sampling_rate, 
        expand_labels=expand_labels,
        batch_size=batch_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(kind == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
