from torch.utils.data import DataLoader
from preprocessing.preprocess import ASIMAnomalyDataset, ASIMAnomalySequence
from pathlib import Path

    
def build_data_pipeline(batch_size: int,
                       data_dir: Path,
                       sub: str,
                       kind: str,
                       window_size: int,
                       stride: int,
                       num_workers: int,
                       sampling_rate: int,  # Add this parameter
                       expand_labels: bool = False) -> DataLoader:
    """Build data pipeline."""
    dataset = ASIMAnomalyDataset(
        kind=kind,
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        sub=sub,
        sampling_rate=sampling_rate, 
        expand_labels=expand_labels
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(kind == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
